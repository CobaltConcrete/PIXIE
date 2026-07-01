import os, sys
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import imageio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)
    
    # check env
    if not torch.cuda.is_available():
        print('CUDA is not available! use CPU instead')
    else:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    # load video for animation sequence
    posedata = TestData(args.posepath, iscrop=args.iscrop, body_detector='rcnn')

    #-- run PIXIE
    pixie_cfg.model.use_tex = args.useTex
    pixie = PIXIE(config = pixie_cfg, device=device)
    visualizer = Visualizer(render_size=args.render_size, config = pixie_cfg, device=device, rasterizer_type=args.rasterizer_type)

    # 2. get the pose/expression of given animation sequence
    writer = imageio.get_writer(os.path.join(savefolder, 'animation.gif'), mode='I')
    for i, batch in enumerate(tqdm(posedata, dynamic_ncols=True)):
        if i % 1 == 0:
            util.move_dict_to_device(batch, device)
            batch['image'] = batch['image'].unsqueeze(0)
            batch['image_hd'] = batch['image_hd'].unsqueeze(0)
            data = {
                'body': batch
            }
            param_dict = pixie.encode(data)
            codedict = param_dict['body']
            moderator_weight = param_dict['moderator_weight']
            opdict = pixie.decode(codedict, param_type='body')

            # print SMPL-X parameters
            print(f'\n-- Frame {i:05} SMPL-X parameters:')
            for key, val in codedict.items():
                if isinstance(val, torch.Tensor):
                    print(f'  {key}: shape={list(val.shape)}, mean={val.mean().item():.4f}, std={val.std().item():.4f}')
                else:
                    print(f'  {key}: {val}')
            
            if args.reproject_mesh and args.rasterizer_type=='standard':
                tform = batch['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2)
                original_image = batch['original_image'][None, ...]
                visualizer.recover_position(opdict, batch, tform, original_image)
            visdict = visualizer.render_results(opdict, data['body']['image_hd'], moderator_weight=moderator_weight, overlay=True)
            pose_ref_shape = visdict['color_shape_images'].clone()
            
            pose_ref_shape_np = (pose_ref_shape[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pose_ref_shape_bgr = pose_ref_shape_np[:, :, [2, 1, 0]]
            name = batch['name']
            cv2.imwrite(os.path.join(savefolder, f'{name}_animate_{i:05}.jpg'), pose_ref_shape_bgr)
            writer.append_data(pose_ref_shape_np)

    writer.close()
    print(f'-- please check the results in {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE')

    parser.add_argument('-i', '--inputpath', default='TestSamples/body/woman-in-white-dress-3830468.jpg', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-p', '--posepath', default='TestSamples/animation', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    # rendering option
    parser.add_argument('--render_size', default=1024, type=int,
                        help='image size of renderings' )
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--reproject_mesh', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to reproject the mesh and render it in original image space, \
                            currently only available if rasterizer_type is standard, because pytorch3d does not support non-squared image...\
                            default is False, means use the cropped image and its corresponding results')
    # save
    parser.add_argument('--deca_path', default=None, type=str,
                        help='absolute path of DECA folder, if exists, will return facial details by running DECA\
                        details of DECA: https://github.com/YadiraF/DECA' )
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--uvtex_type', default='SMPLX', type=str,
                        help='texture type to save, can be SMPLX or FLAME')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveGif', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to visualize other views of the output, save as gif' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveParam', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save parameters as pkl file' )
    parser.add_argument('--savePred', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save smplx prediction as pkl file' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())