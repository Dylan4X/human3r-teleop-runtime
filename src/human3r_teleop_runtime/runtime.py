from typing import Any

import torch
from einops import rearrange

from .export import RichWorldCoordinateExporter
from .geometry import get_camera_parameters
from .ops import apply_threshold, log_optimal_transport, nms, unpad_uv
from .preprocess import FastPreprocessor


class Human3RStreamer:
    def __init__(self, model, device="cuda", size=256, use_ttt3r=False, tf32=False):
        self.model = model
        self.device = device
        self.size = size
        self.use_ttt3r = use_ttt3r
        self.img_res = getattr(model, "mhmr_img_res", None)

        if self.device == "cuda" and tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.transport_alpha = torch.tensor(-10.0, device=device)
        self.world_exporter = RichWorldCoordinateExporter(device=device)
        self.preprocessor = FastPreprocessor(
            device=device,
            size=size,
            img_res=self.img_res,
            get_camera_parameters_fn=get_camera_parameters,
        )
        self.nms = nms
        self.apply_threshold = apply_threshold
        self.unpad_uv = unpad_uv
        self.log_optimal_transport = log_optimal_transport
        self.reset()

    def reset(self):
        self.state_feat = None
        self.state_pos = None
        self.init_state_feat = None
        self.mem = None
        self.init_mem = None
        self.last_smpl_tk = None
        self.last_smpl_id = None
        self.max_smpl_id = -1
        self.prev_reset = False
        self.frame_idx = 0

    @torch.inference_mode()
    def push_frame(self, frame_bgr, reset=False, update=True) -> dict[str, Any]:
        model = self.model
        view = self.preprocessor.prepare_view(frame_bgr, frame_idx=self.frame_idx, reset=reset, update=update)
        batch_size = view["img"].shape[0]

        img_mask = view["img_mask"].reshape(-1, batch_size)
        imgs = view["img"].unsqueeze(0)
        shapes = view["true_shape"].unsqueeze(0)

        imgs = imgs.view(-1, *imgs.shape[2:])
        shapes = shapes.view(-1, 2)
        img_masks_flat = img_mask.view(-1)

        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) == 0:
            out = {"frame_idx": self.frame_idx, "pred": None}
            self.frame_idx += 1
            return out

        img_out, img_pos, _ = model._encode_image(selected_imgs, selected_shapes)
        feat_i = img_out[-1]
        pos_i = img_pos

        imgs_mhmr = view["img_mhmr"].unsqueeze(0).view(-1, *view["img_mhmr"].shape[1:])
        selected_imgs_mhmr = imgs_mhmr[img_masks_flat]
        selected_imgs_mhmr = (selected_imgs_mhmr * 0.5 + 0.5 - self.rgb_mean) / self.rgb_std
        feat_mhmr_i = model.backbone(selected_imgs_mhmr)

        n_patch_mhmr = model.bb_token_res
        scores = model.downstream_head.detect_mhmr(feat_mhmr_i)
        scores = rearrange(scores, "b (nh nw) c -> b c nh nw", nh=n_patch_mhmr, nw=n_patch_mhmr)
        scores = self.nms(scores, kernel=3)
        scores = scores.permute((0, 2, 3, 1))

        feat_mhmr_i = rearrange(feat_mhmr_i, "b (nh nw) c -> b nh nw c", nh=n_patch_mhmr, nw=n_patch_mhmr)
        idx = self.apply_threshold(0.3, scores)
        img_id, h_id, w_id = idx[0], idx[1], idx[2]

        feat_central_mhmr = feat_mhmr_i[img_id, h_id, w_id]
        offset = model.downstream_head.mlp_offset(feat_central_mhmr)
        loc = torch.stack([w_id, h_id]).permute(1, 0)
        loc = (loc + 0.5 + offset) * model.bb_patch_size
        smpl_tk_mhmr = feat_central_mhmr.unsqueeze(0)

        n_patch_cut3r = shapes[0] // model.croco_args["patch_size"]
        feat_cut3r_i = rearrange(feat_i, "b (nh nw) c -> b nh nw c", nh=n_patch_cut3r[0], nw=n_patch_cut3r[1])
        pos_cut3r_i = rearrange(pos_i, "b (nh nw) c -> b nh nw c", nh=n_patch_cut3r[0], nw=n_patch_cut3r[1])

        loc_cut3r = self.unpad_uv(loc, model.mhmr_img_res, *shapes[0])
        smpl_uv_cut3r = (loc_cut3r // model.croco_args["patch_size"]).int()
        w_id_cut3r, h_id_cut3r = smpl_uv_cut3r.T

        feat_central_cut3r = feat_cut3r_i[img_id, h_id_cut3r, w_id_cut3r]
        pos_central_cut3r = pos_cut3r_i[img_id, h_id_cut3r, w_id_cut3r]

        smpl_tk_cut3r = feat_central_cut3r.unsqueeze(0)
        smpl_pos_i = pos_central_cut3r.unsqueeze(0)
        smpl_feat_i = model.downstream_head.mlp_fuse(torch.cat([smpl_tk_mhmr, smpl_tk_cut3r], dim=-1))
        n_humans_i = smpl_feat_i.shape[1]

        if self.state_feat is None:
            self.state_feat, self.state_pos = model._init_state(feat_i, pos_i)
            self.mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
            self.init_state_feat = self.state_feat.clone()
            self.init_mem = self.mem.clone()

        if model.pose_head_flag:
            global_img_feat_i = model._get_img_level_feat(feat_i)
            if self.frame_idx == 0 or self.prev_reset:
                pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = model.pose_retriever.inquire(global_img_feat_i, self.mem)
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=self.device, dtype=pos_i.dtype)
        else:
            global_img_feat_i = None
            pose_feat_i = None
            pose_pos_i = None

        new_state_feat, dec, cross_attn_states = model._recurrent_rollout(
            self.state_feat,
            self.state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            smpl_feat_i,
            smpl_pos_i,
            self.init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
            use_ttt3r=self.use_ttt3r,
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = model.pose_retriever.update_mem(self.mem, global_img_feat_i, out_pose_feat_i)

        if n_humans_i > 0:
            head_input = [
                dec[0].float(),
                dec[model.dec_depth * 2 // 4][:, 1:-n_humans_i].float(),
                dec[model.dec_depth * 3 // 4][:, 1:-n_humans_i].float(),
                dec[model.dec_depth][:, :-n_humans_i].float(),
            ]
            smpl_token = dec[model.dec_depth][:, -n_humans_i:].float()
            smpl_token_cat = torch.cat([smpl_token, smpl_tk_mhmr], dim=-1)
        else:
            head_input = [
                dec[0].float(),
                dec[model.dec_depth * 2 // 4][:, 1:].float(),
                dec[model.dec_depth * 3 // 4][:, 1:].float(),
                dec[model.dec_depth].float(),
            ]
            smpl_token = None
            smpl_token_cat = None

        pred = model._downstream_head(
            head_input,
            shapes,
            pos=pos_i,
            n_humans=n_humans_i,
            smpl_token=smpl_token_cat,
        )

        if smpl_token is not None and self.last_smpl_tk is not None:
            cost_mat = -torch.cdist(self.last_smpl_tk, smpl_token, p=2)
            cost_mat = self.log_optimal_transport(cost_mat, alpha=self.transport_alpha, iters=20)
            matches = cost_mat[:, :-1, :-1]
            max0, max1 = matches.max(2), matches.max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = torch.arange(indices0.shape[1], device=self.device)[None] == indices1.gather(1, indices0)
            mutual1 = torch.arange(indices1.shape[1], device=self.device)[None] == indices0.gather(1, indices1)
            zero = matches.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            valid0 = mutual0 & (mscores0 > 0.2)
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
            smpl_id = indices1.new_full(indices1.shape, -1)
            valid_match1 = indices1[valid1]
            if valid_match1.numel() > 0:
                smpl_id[valid1] = self.last_smpl_id.gather(1, valid_match1[None]).flatten()
            num_new_persons = int((~valid1).sum())
            if num_new_persons > 0:
                new_ids = torch.arange(
                    self.max_smpl_id + 1,
                    self.max_smpl_id + 1 + num_new_persons,
                    device=self.device,
                )
                smpl_id[~valid1] = new_ids
                self.max_smpl_id += num_new_persons
            num_miss_match0 = int((~valid0).sum())
            if num_miss_match0 > 0:
                self.last_smpl_id = torch.cat([smpl_id, self.last_smpl_id[~valid0][None]], dim=1)
                self.last_smpl_tk = torch.cat([smpl_token, self.last_smpl_tk[~valid0][None]], dim=1)
            else:
                self.last_smpl_id = smpl_id.clone()
                self.last_smpl_tk = smpl_token.clone()
        else:
            if smpl_token is not None:
                smpl_id = torch.arange(n_humans_i, device=self.device)[None]
                self.max_smpl_id = max(self.max_smpl_id, n_humans_i - 1)
                self.last_smpl_tk = smpl_token.clone()
                self.last_smpl_id = smpl_id.clone()
            else:
                smpl_id = None

        if smpl_id is not None:
            pred["smpl_id"] = smpl_id

        update_flag = view.get("update", None)
        update_mask = (view["img_mask"] & update_flag) if update_flag is not None else view["img_mask"]
        update_mask = update_mask[:, None, None].float()

        if self.use_ttt3r and self.frame_idx != 0 and not self.prev_reset:
            cat_states = torch.cat(cross_attn_states, dim=0)
            cat_states = rearrange(cat_states, "l h nstate nimg -> 1 nstate nimg (l h)").mean(dim=(-1, -2))
            update_mask_state = update_mask * torch.sigmoid(cat_states)[..., None]
        else:
            update_mask_state = update_mask

        self.state_feat = new_state_feat * update_mask_state + self.state_feat * (1 - update_mask_state)
        self.mem = new_mem * update_mask + self.mem * (1 - update_mask)

        reset_mask = view["reset"]
        if reset_mask is not None:
            reset_mask_f = reset_mask[:, None, None].float()
            self.state_feat = self.init_state_feat * reset_mask_f + self.state_feat * (1 - reset_mask_f)
            self.mem = self.init_mem * reset_mask_f + self.mem * (1 - reset_mask_f)

        self.prev_reset = bool(view["reset"].item())
        out = {"frame_idx": self.frame_idx, "pred": pred}
        self.frame_idx += 1
        return out

    @torch.inference_mode()
    def push_frame_and_export_world(self, frame_bgr, reset=False, update=True) -> dict[str, Any]:
        stream_out = self.push_frame(frame_bgr, reset=reset, update=update)
        return self.world_exporter.export(frame_idx=stream_out["frame_idx"], pred=stream_out["pred"])
