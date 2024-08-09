import torch
import numpy as np

#####################################################
# This coding is from the project (UDA-Animal-Pose) #
#####################################################


class UDAAP(object):
    def __init__(self):
        pass

    @classmethod
    def final_preds(cls, output, center, scale, res):
        coords = cls._get_preds(cls, output)
        preds = coords.clone()

        for i in range(coords.size(0)):
            preds[i] = cls._transform_preds(cls, coords[i], center[i], scale[i], res)

        if preds.dim() < 3:
            preds = preds.view(1, preds.size())

        return preds


    @classmethod
    def transform(cls, pt, center, scale, res, invert=0, rot=0):
        t = cls._get_transform(cls, center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2].astype(int) + 1


    @classmethod
    def im_to_numpy(cls, img):
        img = cls._to_numpy(cls, img)
        img = np.transpose(img, (1, 2, 0))  # H*W*C
        img = np.ascontiguousarray(img)
        return img


    @classmethod
    def im_to_torch(cls, img):
        img = np.transpose(img, (2, 0, 1))  # C*H*W
        img = np.ascontiguousarray(img)
        img = cls._to_torch(cls, img).float()
        if img.max() > 1:
            img /= 255
        return img

    def _to_numpy(self, tensor):
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        elif type(tensor).__module__ != 'numpy':
            raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
        return tensor

    def _to_torch(self, ndarray):
        if type(ndarray).__module__ == 'numpy':
            return torch.from_numpy(ndarray)
        elif not torch.is_tensor(ndarray):
            raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
        return ndarray

    def _get_preds(self, scores):
        ''' get predictions from score maps in torch Tensor
            return type: torch.LongTensor
        '''
        assert scores.dim() == 4, 'Score maps should be 4-dim'
        maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)
        maxval = maxval.view(scores.size(0), scores.size(1), 1)
        idx = idx.view(scores.size(0), scores.size(1), 1) + 1
        preds = idx.repeat(1, 1, 2).float()
        preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1
        pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
        preds *= pred_mask
        return preds

    def _get_transform(self, center, scale, res, rot=0):
        """
        General image processing functions
        """
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1

        if not rot == 0:
            rot = -rot
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1

            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1]/2
            t_mat[1, 2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def _transform_preds(self, coords, center, scale, res):
        for p in range(coords.size(0)):
            coords_trans = self.transform(coords[p, 0:2], center, scale, res, 1, 0)
            coords[p, 0:2] = self._to_torch(self, coords_trans)
        return coords
