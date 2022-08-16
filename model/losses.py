from einops import rearrange
import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F


def manual_cross_entropy(
    labels: torch.Tensor, logits: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    ce = -weight * torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    return torch.mean(ce)

class DetConBLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = torch.tensor(temperature)

    def forward(
        self,
        pred1: torch.Tensor,
        pred2: torch.Tensor,
        target1: torch.Tensor,
        target2: torch.Tensor,
        pind1: torch.Tensor,
        pind2: torch.Tensor,
        tind1: torch.Tensor,
        tind2: torch.Tensor,
        local_negatives: bool = True,
    ) -> torch.Tensor:
        """Compute the NCE scores from pairs of predictions and targets.
        This implements the batched form of the loss described in
        Section 3.1, Equation 3 in https://arxiv.org/pdf/2103.10957.pdf.
        Args:
            pred1: (b, num_samples, d) the prediction from first view.
            pred2: (b, num_samples, d) the prediction from second view.
            target1: (b, num_samples, d) the projection from first view.
            target2: (b, num_samples, d) the projection from second view.
            pind1: (b, num_samples) mask indices for first view's prediction.
            pind2: (b, num_samples) mask indices for second view's prediction.
            tind1: (b, num_samples) mask indices for first view's projection.
            tind2: (b, num_samples) mask indices for second view's projection.
            temperature: (float) the temperature to use for the NCE loss.
            local_negatives (bool): whether to include local negatives
        Returns:
            A single scalar loss for the XT-NCE objective.
        """


        batch_size, num_rois, feature_dim = pred1.shape
        infinity_proxy = 1e9  # Used for masks to proxy a very large number.
        eps = 1e-11

        # Given two indices of (N,), we return the N x N agreement matrix between
        # the two indices (i.e. mask)
        def make_same_obj(ind_0, ind_1):
            same_obj = torch.eq(
                ind_0.reshape([batch_size, num_rois, 1]),
                ind_1.reshape([batch_size, 1, num_rois])
            )
            same_obj = same_obj.unsqueeze(2).to(torch.float)
            return same_obj


        same_obj_aa = make_same_obj(pind1, tind1)
        same_obj_ab = make_same_obj(pind1, tind2)
        same_obj_ba = make_same_obj(pind2, tind1)
        same_obj_bb = make_same_obj(pind2, tind2)

        # L2 normalize the tensors to use for the cosine-similarity
        pred1 = F.normalize(pred1, dim=-1)
        pred2 = F.normalize(pred2, dim=-1)
        target1 = F.normalize(target1, dim=-1)
        target2 = F.normalize(target2, dim=-1)

        device_id = torch.cuda.current_device()
        curr_rank = torch.distributed.get_rank()
        if torch.distributed.get_world_size() > 1:
            # Grab tensor across GPUs
            all_target1 = concat_all_gather(target1)
            all_target1[curr_rank] = target1
            all_target1 = torch.cat(all_target1, dim=0)

            all_target2 = concat_all_gather(target2)
            all_target2[curr_rank]  = target2
            all_target2 = torch.cat(all_target2, dim=0)

            # Create the labels by using the current device ID and offsetting.
            labels_idx = torch.arange(batch_size) + curr_rank * batch_size
            labels_idx = labels_idx.type(torch.LongTensor)
            enlarged_batch_size = all_target1.shape[0]
            labels = F.one_hot(labels_idx, num_classes=enlarged_batch_size).to(device_id)
        else:
            all_target1 = target1
            all_target2 = target2
            labels = F.one_hot(torch.arange(batch_size), num_classes=batch_size).to(device_id)
        
        # import ipdb
        # ipdb.set_trace()

        labels = labels.unsqueeze(dim=2).unsqueeze(dim=1)

        # Do our matmuls and mask out appropriately.
        logits_aa = torch.einsum("abk,uvk->abuv", pred1, all_target1) / (
            self.temperature + eps
        )
        logits_bb = torch.einsum("abk,uvk->abuv", pred2, all_target2) / (
            self.temperature + eps
        )
        logits_ab = torch.einsum("abk,uvk->abuv", pred1, all_target2) / (
            self.temperature + eps
        )
        logits_ba = torch.einsum("abk,uvk->abuv", pred2, all_target1) / (
            self.temperature + eps
        )

        labels_aa = labels * same_obj_aa
        labels_ab = labels * same_obj_ab
        labels_ba = labels * same_obj_ba
        labels_bb = labels * same_obj_bb

        logits_aa = logits_aa - infinity_proxy * labels * same_obj_aa
        logits_bb = logits_bb - infinity_proxy * labels * same_obj_bb
        labels_aa = 0.0 * labels_aa
        labels_bb = 0.0 * labels_bb

        if not local_negatives:
            logits_aa = logits_aa - infinity_proxy * labels * (1 - same_obj_aa)
            logits_ab = logits_ab - infinity_proxy * labels * (1 - same_obj_ab)
            logits_ba = logits_ba - infinity_proxy * labels * (1 - same_obj_ba)
            logits_bb = logits_bb - infinity_proxy * labels * (1 - same_obj_bb)

        labels_abaa = torch.cat([labels_ab, labels_aa], dim=2)
        labels_babb = torch.cat([labels_ba, labels_bb], dim=2)

        labels_0 = labels_abaa.reshape((batch_size, num_rois, -1))
        labels_1 = labels_babb.reshape((batch_size, num_rois, -1))

        num_positives_0 = torch.sum(labels_0, dim=-1, keepdim=True)
        num_positives_1 = torch.sum(labels_1, dim=-1, keepdim=True)

        labels_0 = labels_0 / torch.maximum(num_positives_0, torch.tensor(1.0))
        labels_1 = labels_1 / torch.maximum(num_positives_1, torch.tensor(1.0))

        obj_area_0 = torch.sum(make_same_obj(pind1, pind1), dim=(2, 3))
        obj_area_1 = torch.sum(make_same_obj(pind2, pind2), dim=(2, 3))

        weights_0 = torch.greater(num_positives_0[..., 0], 1e-3).to(torch.float)
        weights_0 = weights_0 / obj_area_0
        weights_1 = torch.greater(num_positives_1[..., 0], 1e-3).to(torch.float)
        weights_1 = weights_1 / obj_area_1

        logits_abaa = torch.cat([logits_ab, logits_aa], dim=2)
        logits_babb = torch.cat([logits_ba, logits_bb], dim=2)

        logits_abaa = logits_abaa.reshape((batch_size, num_rois, -1))
        logits_babb = logits_babb.reshape((batch_size, num_rois, -1))

        loss_a = manual_cross_entropy(labels_0, logits_abaa, weight=weights_0)
        loss_b = manual_cross_entropy(labels_1, logits_babb, weight=weights_1)
        loss = loss_a + loss_b
        return loss

class VMFClusterLoss(nn.Module):
    def __init__(self,
                min_classes: int = 2,
                max_classes: int = 81,
    ) -> None:
        super().__init__()
        self.min_classes = min_classes
        self.max_classes = max_classes
    
    def _em_cluster(self, flat_ft_map, K):
        flat_ft_map = flat_ft_map.astype(np.float32)
        flat_ft_map = np.ascontiguousarray(flat_ft_map)
        _, d = flat_ft_map.shape

        clus = faiss.Clustering(d, K)

        clus.seed = np.random.randint(1234)
        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        
        index = faiss.IndexFlatL2(d)

        # perform the training
        clus.train(flat_ft_map, index)

        # get assignments
        _, I = index.search(flat_ft_map, 1)

        # return cluster centroids and assignments
        return (faiss.vector_float_to_array(clus.centroids).reshape(K, d), I[:, 0])

    def forward(
        self,
        omap1: torch.Tensor,
        omap2: torch.Tensor,
        tmap1: torch.Tensor,
        tmap2: torch.Tensor,
        vmf_temp: torch.Tensor,
        vmf_weight: float,
    ) -> torch.Tensor:
        eps = 1e-11

        # Render cluster centroids in this batch
        cluster_resolution = np.random.randint(self.min_classes, self.max_classes)
        b, c, h, w = tmap1.shape
        tmap1 = rearrange(tmap1, 'b c h w -> (b h w) c')
        tmap1 = F.normalize(tmap1, dim=-1)
        tmap1 = tmap1.cpu().numpy()
        tmap2 = rearrange(tmap2, 'b c h w -> (b h w) c')
        tmap2 = F.normalize(tmap2, dim=-1)
        tmap2 = tmap2.cpu().numpy()
        
        # centroids are (cluster_resolution, c), labels are (b * h * w,)
        centroids1, labels1 = self._em_cluster(tmap1, cluster_resolution)
        centroids2, labels2 = self._em_cluster(tmap2, cluster_resolution)

        # Tensorize
        labels1 = torch.from_numpy(labels1)
        labels2 = torch.from_numpy(labels2)
        device_id = torch.cuda.current_device()
        labels1 = F.one_hot(labels1, num_classes=cluster_resolution).to(device_id)
        labels2 = F.one_hot(labels2, num_classes=cluster_resolution).to(device_id)

        centroids1 = torch.from_numpy(centroids1).to(torch.float)
        centroids2 = torch.from_numpy(centroids2).to(torch.float)

        # Reshape:
        # labels (b * h * w, cluster_resolution)
        # omap (b, c, h, w) --> (b, h * w, c)
        # centroids (cluster_resolution, c)
        omap1 = rearrange(omap1, 'b c h w -> b (h w) c')
        omap1 = F.normalize(omap1, dim=-1)
        omap2 = rearrange(omap2, 'b c h w -> b (h w) c')
        omap2 = F.normalize(omap2, dim=-1)
        centroids1 = torch.repeat_interleave(centroids1.unsqueeze(0), b, dim=0).to(device_id)
        centroids2 = torch.repeat_interleave(centroids2.unsqueeze(0), b, dim=0).to(device_id)
        labels1 = labels1.view(b, h * w, cluster_resolution)
        labels2 = labels2.view(b, h * w, cluster_resolution)
        ce_weights = torch.ones(b, h * w).to(device_id) * vmf_weight

        # Intra-view loss
        intra_logits1 = -torch.einsum("bpc,bkc->bpk", omap1, centroids1) / (
            vmf_temp + eps
        )
        intra_loss1 = manual_cross_entropy(labels1, intra_logits1, weight=ce_weights)
        intra_logits2 = -torch.einsum("bpc,bkc->bpk", omap2, centroids2) / (
            vmf_temp + eps
        )
        intra_loss2 = manual_cross_entropy(labels2, intra_logits2, weight=ce_weights)

        # Inter-view loss
        inter_logits1 = -torch.einsum("bpc,bkc->bpk", omap1, centroids2) / (
            vmf_temp + eps
        )
        inter_loss1 = manual_cross_entropy(labels1, inter_logits1, weight=ce_weights)
        inter_logits2 = -torch.einsum("bpc,bkc->bpk", omap2, centroids1) / (
            vmf_temp + eps
        )
        inter_loss2 = manual_cross_entropy(labels2, inter_logits2, weight=ce_weights)


        return intra_loss1 + intra_loss2 + inter_loss1 + inter_loss2
        

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    return tensors_gather
