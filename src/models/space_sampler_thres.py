"""Spatial sampler with threshold
"""
import sys

import torch
import numpy as np
from skimage import measure
from torchvision.ops import nms
from scipy.optimize import linear_sum_assignment


class SpatialSamplerThres():
    def __init__(self, top_k, min_b_size, max_b_size, reorder_pair=False,
                 reorder_vid=True, avg_across_time=False, steps=7, iou_thres=0.3,
                 area_limit=0.25):
        """Initialize the spatial sampler.

        Mask will be created from attention by thresholding as:
            mask = alpha * attn.mean() + beta * attn.std()

        Args:
            top_k: (int) number of top regions to sample. If 0 -> not used
            min_b_size: (int) minimum size of each bbox to sample
            max_b_size: (int) maximum size of each bbox to sample
        """
        self.top_k = top_k
        self.min_b_size = min_b_size
        self.max_b_size = max_b_size
        self.steps = steps
        self.iou_thres = iou_thres
        self.area_limit = area_limit
        self.reorder_pair = reorder_pair
        self.reorder_vid = reorder_vid
        self.avg_across_time = avg_across_time

        self._prev_bboxes = None

    def reset(self):
        """Reset _prev_bbox. Use at the start of a frame sequence
        """
        self._prev_bboxes = None

    def sample_frame(self, attn, img_size, reorder_pair=None):
        """Sampling function

        Args:
            attn: attention of the current frame. Shape of (B, C2, H2, W2)
                Can be groundtruth attention or halllucination from prev frame
            img_size: (int) size of a square image frame
            reorder_pair: whether to reorder the bboxes pairwise

        Return:
            results: tensor of shape [B, top_k, 4]. The last dimension defines
                the bounding boxes as (top, left, bottom, right).
                None if top_k = 0
        """
        if reorder_pair is None:
            reorder_pair = self.reorder_pair

        if self.top_k == 0:
            return None

        assert attn.shape[-1] == attn.shape[-2]
        attn_size = attn.shape[-1]
        batch_size = attn.shape[0]
        if self._prev_bboxes is not None:
            assert len(self._prev_bboxes) == batch_size

        # Flatten the attention
        attn = attn.mean(dim=1)

        # For each sample in batch
        results = []
        for b in range(batch_size):
            # Get bboxes in attention plane
            props, top_segids = self._get_bbox_from_attn(
                attn[b].cpu().detach().numpy())

            # Project to image plane
            bboxes = []
            for sid in top_segids:
                top, left, bottom, right = self._project_image_plane(
                    props[sid], attn_size=attn_size, img_size=img_size)
                bboxes.append([top, left, bottom, right])

            # Use center box if nothing is found
            if len(bboxes) == 0:
                bboxes.append([(img_size - self.max_b_size)//2,
                               (img_size - self.max_b_size)//2,
                               (img_size - self.max_b_size)//2 + self.max_b_size,
                               (img_size - self.max_b_size)//2 + self.max_b_size,
                               ])

            # Append the last item if not enough bboxes
            if len(bboxes) != self.top_k:
                bboxes += [bboxes[-1]] * (self.top_k - len(bboxes))

            # Reorder
            if reorder_pair:
                if (self._prev_bboxes is not None) and (self._prev_bboxes[b] is not None):
                    bboxes = self._sort_bboxes_pair(self._prev_bboxes[b], bboxes)
                else:
                    if self._prev_bboxes is None:
                        self._prev_bboxes = [None for _ in range(batch_size)]
                    self._prev_bboxes[b] = bboxes

            # Collect results
            results.append(bboxes)

        # Update prev bboxes
        self._prev_bboxes = np.copy(results)
        return np.array(results)

    def sample_multiple_frames(self, attns, img_size, reorder_vid=None,
                               reorder_pair=None, avg_across_time=None):
        """Wrapper of sample_frame for multiple frames

        Args:
            attn: attention of all frames. Shape of (B, T, C2, H2, W2)
                Can be groundtruth attention or halllucination from prev frame
            img_size: (int) size of a square image frame
            reorder: whether to reorder the bboxes
            avg_across_time: whether to average the bbox size across time.
                `reorder` must be True to use `avg_across_time`

        Return:
            results: tensor of shape [B, T, top_k, 4]. The last dimension defines
                the bounding boxes as (top, left, bottom, right).
                None if top_k = 0
        """
        if reorder_vid is None:
            reorder_vid = self.reorder_vid
        if reorder_pair is None:
            reorder_pair = self.reorder_pair
        if avg_across_time is None:
            avg_across_time = self.avg_across_time

        if self.top_k == 0:
            return None

        if avg_across_time is True:
            assert reorder_pair is True or reorder_vid is True, \
                '`reorder_pair` or `reorder_vid` must be True to use `avg_across_time`'

        n_frames = attns.shape[1]
        all_bboxes = []

        self.reset()
        for t in range(n_frames):
            bboxes = self.sample_frame(attns[:, t], img_size, reorder_pair)
            all_bboxes.append(np.expand_dims(bboxes, axis=1))
        all_bboxes = np.concatenate(all_bboxes, axis=1)

        # Get the average across time -----------------------------------------
        if avg_across_time:
            # Find the average heights and widths
            heights = all_bboxes[:, :, :, 2] - all_bboxes[:, :, :, 0]
            widths = all_bboxes[:, :, :, 3] - all_bboxes[:, :, :, 1]
            avg_heights = np.round(heights.mean(axis=1, keepdims=True)).astype(int)
            avg_widths = np.round(widths.mean(axis=1, keepdims=True)).astype(int)

            # Repeat to broadcast
            avg_heights = np.repeat(avg_heights, n_frames, axis=1)
            avg_widths = np.repeat(avg_widths, n_frames, axis=1)

            # Find the centers
            y_centers = 0.5*(all_bboxes[:, :, :, 2] + all_bboxes[:, :, :, 0])
            x_centers = 0.5*(all_bboxes[:, :, :, 3] + all_bboxes[:, :, :, 1])

            # Get new top, left, bottom, right
            new_tops = y_centers - (avg_heights // 2)
            new_bottoms = y_centers + (avg_heights // 2)
            new_lefts = x_centers - (avg_widths // 2)
            new_rights = x_centers + (avg_widths // 2)

            # Adjust
            delta = (-new_tops) * (new_tops < 0)
            new_tops += delta
            new_bottoms += delta

            delta = (new_bottoms-img_size) * (new_bottoms > img_size)
            new_tops -= delta
            new_bottoms -= delta

            delta = (-new_lefts) * (new_lefts < 0)
            new_lefts += delta
            new_rights += delta

            delta = (new_rights-img_size) * (new_rights > img_size)
            new_lefts -= delta
            new_rights -= delta

            # Collect new bboxes
            all_bboxes = np.stack([new_tops, new_lefts, new_bottoms, new_rights],
                                  axis=3).astype(int)

        # Reorder multi frames using dijkstra ---------------------------------
        if reorder_vid:
            for i in range(len(all_bboxes)):
                all_bboxes[i], _ = self._sort_bboxes_dijkstra(all_bboxes[i])
        return all_bboxes

    # def _get_bbox_from_attn(self, attn, simple_return=True):
    #     """Segment and get bbox from attention map

    #     Args:
    #         attn: attention map
    #         simple_return: (boolean) If True, return only props and top_segids.
    #             Otherwise, return also mask, segments, and scores

    #     Return:
    #         props: list of RegionProperties
    #         top_segids: list of segment id of the top segments based on scores
    #         mask: binary mask of the attention map after thresholding
    #         segments: labeled segments generated from mask
    #         scores: array of scores wrt each segment
    #     """
    #     # Mask the attention by thresholding
    #     thres = self.alpha*attn.mean() + self.beta*attn.std()
    #     mask = np.where(attn > thres, 1.0, 0.0)

    #     # Segment the mask, each segment will be assigned a label
    #     segments, n_seg = measure.label(mask, return_num=True)

    #     # Find bounding boxes
    #     props = measure.regionprops(segments)

    #     # Find the top segments
    #     scores = np.zeros(n_seg)
    #     for i, prop in enumerate(props):
    #         scores[i] = attn[prop.coords[:, 0], prop.coords[:, 1]].sum()
    #     top_segids = scores.argsort()[::-1][:self.top_k]

    #     if simple_return:
    #         return props, top_segids
    #     return props, top_segids, mask, segments, scores

    def _get_bbox_from_attn(self, attn):
        """Segment and get bbox from attention map

        Args:
            attn: attention map
            simple_return: (boolean) If True, return only props and top_segids.
                Otherwise, return also mask, segments, and scores

        Return:
            props: list of RegionProperties
            top_segids: list of segment id of the top segments based on scores
        """
        thres_lst = attn.mean() + np.linspace(-attn.std(), attn.std(), self.steps)

        props = []
        for thres in thres_lst:
            mask = np.where(attn > thres, 1.0, 0.0)
            segments, n_seg = measure.label(mask, return_num=True)
            props += [x for x in measure.regionprops(segments)
                      if len(x.coords) < self.area_limit*np.prod(attn.shape)]

        bboxes = [x.bbox for x in props]
        scores = [(attn[x.coords[:, 0], x.coords[:, 1]]).sum() for x in props]

        top_ids = nms(torch.tensor(bboxes, dtype=torch.float32),
                      torch.tensor(scores, dtype=torch.float32),
                      self.iou_thres)[:self.top_k]

        return props, top_ids

    def _project_image_plane(self, prop, attn_size, img_size):
        """Project a bounding box to image plane

        Args:
            prop: (RegionProperties) property of a region

        Return:
            top, left, bottom, right: corner positions of the bbox in image plane
        """
        # Get the scale from original image size to the current attention size
        scale = img_size / attn_size

        # Get the square bbox in attn plane
        min_row, min_col, max_row, max_col = prop.bbox
        b_h, b_w = max_row-min_row, max_col-min_col
        b_size = max(b_h, b_w)
        b_center = ((max_row+min_row)//2, (max_col+min_col)//2)

        # Convert to the scale of image plane
        b_center_img = (int(b_center[0]*scale), int(b_center[1]*scale))
        b_size_img = np.clip(int(b_size*scale), self.min_b_size, self.max_b_size)

        top = b_center_img[0] - b_size_img//2
        left = b_center_img[1] - b_size_img//2
        bottom = b_center_img[0] + b_size_img//2
        right = b_center_img[1] + b_size_img//2

        # Adjust the bbox to get a square one
        if top < 0:
            bottom -= top
            top -= top
        if left < 0:
            right -= left
            left -= left
        if bottom > img_size:
            top -= (bottom - img_size)
            bottom -= (bottom - img_size)
        if right > img_size:
            left -= (right - img_size)
            right -= (right - img_size)

        assert (bottom-top == b_size_img) and (right-left == b_size_img), \
            'Bounding box out of bound: {}, {}, {}, {}'.format(
                top, left, bottom, right)

        return top, left, bottom, right

    def _sort_bboxes_pair(self, prev_bboxes, bboxes, use_center=False):
        """Sort bboxes by the min distance wrt the previous bboxes using
        Hungarian algorithm. Each bbox is represented as a tuple of 4 numbers:
        (top, left, bottom, right)

        Args:
            prev_bboxes: list of bboxes in the previous frame
            bboxes: list of bboxes in the current frame

        Return:
            Sorted list of bboxes
        """
        M, N = len(prev_bboxes), len(bboxes)
        assert M == N, 'Number of bboxes does not match: {}, {}'.format(M, N)

        cost = np.zeros((M, N), dtype=np.float32)
        if use_center:
            prev_centers = [((t+b)//2, (l+r)//2) for t, l, b, r in prev_bboxes]
            centers = [((t+b)//2, (l+r)//2) for t, l, b, r in bboxes]

            for i in range(M):
                for j in range(N):
                    p1 = prev_centers[i]
                    p2 = centers[j]
                    cost[i, j] = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        else:
            for i in range(M):
                for j in range(N):
                    box1 = prev_bboxes[i]
                    box2 = bboxes[j]
                    cost[i, j] = sum([(box1[k]-box2[k])**2 for k in range(4)])

        # Matching using Hungarian algorithm
        _, col_ind = linear_sum_assignment(cost)

        return [bboxes[i] for i in col_ind]

    def _sort_bboxes_dijkstra(self, bboxes):
        T, K, _ = bboxes.shape
        visited = np.zeros([T, K], dtype=bool)
        orders = []

        for _ in range(K-1):
            graph = BBGraph(bboxes, visited)
            min_dist, min_prev, min_pair = sys.maxsize, None, None

            # Run dijkstra for each bbox of the first frame
            for src in range(K):
                dist, prev = graph.dijkstra(src)

                # Find the min path to the bboxes of the last frame
                for target in range(graph.n_vertices-K, graph.n_vertices):
                    if min_dist > dist[target]:
                        min_dist = dist[target]
                        min_prev = prev
                        min_pair = (src, target)

            # Retrieve the path with the lowest cost first
            path = graph.get_path(min_prev, min_pair[0], min_pair[1])
            orders.append(path)

            # Mark the path as visited to ignore
            for t in range(T):
                visited[t, path[t]] = True

        # The last remaining path
        path = [np.argmin(x) for x in visited]
        orders.append(path)

        # Swap the bboxes based on the new order
        orders = np.stack(orders).T
        new_bboxes = np.array([bboxes[t][orders[t]] for t in range(T)])
        return new_bboxes, orders

    def get_regions_from_bboxes(self, x, bboxes):
        """Get regions from input, given the bounding boxes

        Args:
            x: input tensor of shape (B, T, C, H, W)

        Return:
            regions: list of K tensors, each of shape (B, T, C, H', W')
        """
        batch_size, num_segments, _, _, _ = x.shape

        regions = []
        for k in range(self.top_k):
            regions_k = []
            for b in range(batch_size):
                tops = bboxes[b, :, k, 0]
                lefts = bboxes[b, :, k, 1]
                bottoms = bboxes[b, :, k, 2]
                rights = bboxes[b, :, k, 3]

                # Batch regions across time b/c of consisting size
                regions_k_b = []
                for t in range(num_segments):
                    regions_k_b.append(
                        x[b, t, :, tops[t]:bottoms[t], lefts[t]:rights[t]
                          ].unsqueeze(dim=0))
                regions_k_b = torch.cat(regions_k_b, dim=0)
                regions_k.append(regions_k_b.unsqueeze(dim=0))
            regions_k = torch.cat(regions_k, dim=0)
            regions.append(regions_k)
        return regions


class BBGraph:
    """Graph for bboxes"""
    def __init__(self, bboxes, visited):
        """
        Args:
            bboxes: ndarray of shape (T, K, 4). The bboxes for T frames. Each
                frame has K bboxes. Each bbox is defined by 4 numbers:
                (top, left, bottom, right). The generated graph is an adjacency
                matrix of shape (T*K, T*K)
            visited: boolean ndarray of shape (T, K). True if a bbox is already
                used in another path. The graph is only constructed from
                unvisited nodes
        """
        T = bboxes.shape[0]
        K = bboxes.shape[1]
        self.top_k = K
        self.n_vertices = T*K
        self.graph = np.zeros([self.n_vertices, self.n_vertices])

        # Make adjacency graph from bbox. An edge is the cost to move from a
        # bbox of frame t to a bbox of frame t+1
        for t in range(T-1):
            frame_1 = bboxes[t]
            frame_2 = bboxes[t+1]

            # For each unvisited bbox (node) of frame t
            for k1 in range(K):
                if visited[t, k1]:
                    continue
                t1, l1, b1, r1 = frame_1[k1]

                # Find the cost to each unvisited (bbox) of frame t+1
                for k2 in range(K):
                    if visited[t, k2]:
                        continue
                    t2, l2, b2, r2 = frame_2[k2]

                    # Convert from trellis index (bbox) to vertex index (graph)
                    v1 = t*K + k1
                    v2 = (t+1)*K + k2

                    # Compute score between 2 vertices
                    area1 = (b1-t1)*(r1-l1)  # area of bbox1
                    area2 = (b2-t2)*(r2-l2)  # area of bbox2
                    d2 = max(0.1, ((frame_1[k1]-frame_2[k2])**2).sum())  # square distance
                    self.graph[v1, v2] = d2 + abs(area1-area2)

    def _get_min_index(self, dist, visited):
        """Get the unvisited vertex index with the minimum distance. Return None
        if cannot find any.
        """
        min_dst = sys.maxsize
        min_index = None

        for v in range(self.n_vertices):
            if dist[v] < min_dst and not visited[v]:
                min_dst = dist[v]
                min_index = v
        return min_index

    def dijkstra(self, src):
        """Dijkstra algorithm to find the shortest path from src to all other
        nodes

        Args:
            src: index of the src node to start from

        Return:
            dist: list of distance from the given src node to all other nodes
            prev: the list of previous node to trace back a path
        """
        dist = [sys.maxsize for _ in range(self.n_vertices)]
        dist[src] = 0
        visited = [False for _ in range(self.n_vertices)]
        prev = [None for _ in range(self.n_vertices)]

        for _ in range(self.n_vertices):
            u = self._get_min_index(dist, visited)
            if u is None:
                continue

            visited[u] = True
            for v in range(self.n_vertices):
                alt_dst = dist[u] + self.graph[u][v]
                if (self.graph[u, v] > 0) and (not visited[v]) and (dist[v] > alt_dst):
                    dist[v] = alt_dst
                    prev[v] = u
        return dist, prev

    def get_path(self, prev, src, target):
        """Trace back the path from prev list. Return the path of bbox index
        per frame (not vertex index)
        """
        path = []
        u = target
        while prev[u] is not None:
            path.append(u)
            u = prev[u]
        path.append(src)
        return [x % self.top_k for x in path[::-1]]  # reverse the path and convert index


if __name__ == '__main__':
    """Test the implementation"""
    from time import time

    spatial_sampler = SpatialSamplerThres(
        top_k=3, min_b_size=64, max_b_size=112, alpha=1.0, beta=0.1)

    batch = 5
    length = 10
    img_size = 224
    x = torch.rand((batch, length, 3, img_size, img_size), dtype=torch.float32).cuda()
    attn = torch.rand([batch, length, 64, 14, 14], dtype=torch.float32).cuda()

    spatial_sampler.reset()
    for t in range(length):
        st = time()
        results = spatial_sampler.sample_frame(attn[:, t], img_size)
        assert results.shape == (batch, 3, 4)
        print(time() - st)

    print('------------------------------------------------------------------')
    st = time()
    spatial_sampler.sample_multiple_frames(attn, img_size)
    print(time() - st)
