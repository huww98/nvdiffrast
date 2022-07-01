import unittest

import torch

import nvdiffrast.torch as dr

class AATest(unittest.TestCase):
    def setUp(self):
        DEVICE = torch.device('cuda:0')
        self.pos = torch.tensor([[
            [-2,  6, 0.2, 1],
            [-2, -6, 0.2, 1],
            [ 4,  0, 0.2, 1],
            [ 0,  4, 0  , 1],
            [ 0, -4, 0  , 1],
            [ 4,  0, 0  , 1],
        ]], dtype=torch.float32, device=DEVICE)
        self.tri = torch.tensor([
            [0,1,2],
            [3,4,5],
        ], dtype=torch.int32, device=DEVICE)
        self.color = torch.tensor([[
            [[0,0], [0,1]],
        ]], dtype=torch.float32, device=DEVICE)

    def test_aligned(self):
        self.pos.requires_grad = True
        ctx = dr.RasterizeGLContext(output_db=False)
        rast, _ = dr.rasterize(ctx, self.pos, self.tri, resolution=(1, 2))
        color, bg_subpixel = dr.antialias(self.color, rast, self.pos, self.tri, bg_subpixel=True)
        torch.testing.assert_allclose(color, self.color)
        self.assertEqual(bg_subpixel.item(), 0)

        color.sum().backward()
        grad_expected = torch.zeros_like(self.pos)
        grad_expected[0, 3:5, 0] = -0.5
        torch.testing.assert_allclose(self.pos.grad, grad_expected)

    def test_blend(self):
        self.pos[0, 3:5, 0] = -0.2
        self.pos.requires_grad = True
        ctx = dr.RasterizeGLContext(output_db=False)

        for d_bg_subpixel in [True, False]:
            # This flag should have no inpact here
            with self.subTest(d_bg_subpixel=d_bg_subpixel):
                self.pos.grad = None
                rast, _ = dr.rasterize(ctx, self.pos, self.tri, resolution=(1, 2))
                color, bg_subpixel = dr.antialias(self.color, rast, self.pos, self.tri, bg_subpixel=True)
                color_expected = torch.tensor([[
                    [[0,0.2], [0,1]],
                ]], dtype=torch.float32, device=color.device)
                torch.testing.assert_allclose(color, color_expected)
                self.assertEqual(bg_subpixel.item(), 0)

                loss = color.sum()
                if d_bg_subpixel:
                    loss = loss + bg_subpixel.sum()
                loss.backward()
                grad_expected = torch.zeros_like(self.pos)
                grad_expected[0, 3:5, 0] = -0.5
                grad_expected[0, 3:5, 3] = -0.1
                torch.testing.assert_allclose(self.pos.grad, grad_expected)

    def test_bg_subpixel(self):
        pos = self.pos[:, 3:].contiguous()
        tri = self.tri[1:] - 3
        pos[0, :2, 0] = -0.2

        ctx = dr.RasterizeGLContext(output_db=False)

        for flip in [-1, 1]:
            with self.subTest(flip=flip):
                pos_f = pos.clone()
                pos_f[..., 0] *= flip
                pos_f.requires_grad = True

                rast, _ = dr.rasterize(ctx, pos_f, tri, resolution=(1, 2))
                color, bg_subpixel = dr.antialias(self.color, rast, pos_f, tri, bg_subpixel=True)
                torch.testing.assert_allclose(bg_subpixel.cpu(), [0.2])

                bg_subpixel.sum().backward()
                grad_expected = torch.zeros_like(pos_f)
                grad_expected[0, :2, 0] = -0.5 * flip
                grad_expected[0, :2, 3] = -0.1
                torch.testing.assert_allclose(pos_f.grad, grad_expected)

    def test_mesh_border_no_grad(self):
        pos = self.pos
        tri = self.tri

        pos.requires_grad = True
        ctx = dr.RasterizeGLContext(output_db=False)
        rast, _ = dr.rasterize(ctx, pos, tri, resolution=(1, 2))
        color = dr.antialias(self.color, rast, pos, tri, mesh_border=False)

        color.sum().backward()
        grad_expected = torch.zeros_like(pos)
        torch.testing.assert_allclose(pos.grad, grad_expected)


    def test_mesh_border(self):
        pos = torch.cat((self.pos, self.pos[:, -1:]), dim=1)
        tri = torch.cat((self.tri, torch.tensor([
            [4,3,6],
        ], dtype=torch.int32, device=self.tri.device)), dim=0)

        pos.requires_grad = True
        ctx = dr.RasterizeGLContext(output_db=False)
        rast, _ = dr.rasterize(ctx, pos, tri, resolution=(1, 2))
        color = dr.antialias(self.color, rast, pos, tri, mesh_border=False)

        color.sum().backward()
        grad_expected = torch.zeros_like(pos)
        grad_expected[0, 3:5, 0] = -0.5
        torch.testing.assert_allclose(pos.grad, grad_expected)
