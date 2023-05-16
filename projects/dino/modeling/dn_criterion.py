# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from detrex.utils import get_world_size, is_dist_avail_and_initialized

from .two_stage_criterion import TwoStageCriterion
from detrex.layers import box_cxcywh_to_xyxy
import torch.nn.functional as F
import random


class DINOCriterion(TwoStageCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def forward(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = super(DINOCriterion, self).forward(outputs, targets)
        # import pdb;pdb.set_trace()
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        aux_num = 0
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)

        # Loss is a dictionary, we want to add the edge loss here
        # Calculate edge losses
        # For each target image
        # start with tensor incase loss is 0, then can't detach
        edge_loss_total = torch.tensor(0, dtype=torch.float, device=outputs["pred_boxes"].device)
        for index in range(len(targets)):
            # (height, width)
            #breakpoint()
            target_edge = targets[index]["edges"]
            image_size = targets[index]["image_size"]
            # (num_predictions=900, 4)
            pred_boxes = outputs['pred_boxes'][index]
            w, h = image_size[1], image_size[0]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float).to(pred_boxes.device)
            # pick a random box
            box_index = random.randrange(len(pred_boxes))
            xmin, ymin, xmax, ymax = box_cxcywh_to_xyxy(pred_boxes[box_index]) * image_size_xyxy
            # convert to (xmin, ymin, xmax, ymax)

            def convert(num):
                return torch.round(num).to(torch.int32)

            xmin, ymin, xmax, ymax = convert(xmin), convert(ymin), convert(xmax), convert(ymax)
            xmin, ymin = torch.max(xmin, torch.tensor(0)), torch.max(ymin, torch.tensor(0))
            xmax, ymax = torch.min(torch.tensor(image_size[1]), xmax), torch.min(torch.tensor(image_size[0]), ymax)
            # ensure within image bounds

            edge_in_box = torch.any(target_edge[ymin:ymax, xmin:xmax])
            # If there is no edge in the box, we find the closest edge and use that as the loss
            if edge_in_box == 0:
                idx = torch.argwhere(target_edge)
                pred_row, pred_col = (ymin + ymax) // 2, (xmin + xmax) // 2
                pred = torch.tensor((pred_row, pred_col)).to(pred_boxes.device)
                nearest_row, nearest_col = idx[((idx - pred) ** 2).sum(1).argmin()]
                #print("nearest row:", nearest_row, "nearest col", nearest_col)
                #print("pred row:", pred_row, "pred col:", pred_col)
                nearest_row_norm, nearest_col_norm = nearest_row / h, nearest_col / w
                target_pixel = torch.as_tensor([nearest_col_norm, nearest_row_norm]).to(pred_boxes.device)
                edge_loss = F.l1_loss(pred_boxes[box_index][:2], target_pixel, reduction="mean")
                #print("edge loss", edge_loss)
                edge_loss_total += edge_loss
                #breakpoint()
        #breakpoint()
        # needs to be float to multiply by weight_dict
        losses["edge_loss"] = edge_loss_total.float()
        #breakpoint()
        return losses


    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = (
                dn_metas["output_known_lbs_bboxes"],
                dn_metas["dn_num"],
                dn_metas["single_padding"],
            )
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["labels"])).long().cuda()
                    t = t.unsqueeze(0).repeat(dn_num, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(dn_num)) * single_padding
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(
                    self.get_loss(
                        loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num, **kwargs
                    )
                )

            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")

        for i in range(aux_num):
            # dn aux loss
            l_dict = {}
            if dn_metas and "output_known_lbs_bboxes" in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes["aux_outputs"][i]
                for loss in self.losses:
                    kwargs = {}
                    if "labels" in loss:
                        kwargs = {"log": False}
                    l_dict.update(
                        self.get_loss(
                            loss,
                            output_known_lbs_bboxes_aux,
                            targets,
                            dn_idx,
                            num_boxes * dn_num,
                            **kwargs,
                        )
                    )
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
            else:
                l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses
