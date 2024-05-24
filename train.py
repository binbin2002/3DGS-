#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    '''
    1.每迭代1000次，增加SH的阶数
    2.随机选择一个相机视角
    3.渲染图像，获取视点空间点张量，可见性过滤器，半径等信息
    4.计算损失L1和SSIM
    5.通过无梯度的上下文进行后续操作：
        1.根据迭代次数进行点云密度操作
        2.记录最大半径，添加稠密化统计信息
        3.根据条件进行点云密度增加和修剪
        4.根据条件重置不透明度
        5.进行优化器步骤的参数更新'''
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset) # 准备TensorBoard输出和记录器
    gaussians = GaussianModel(dataset.sh_degree) # 创建高斯模型
    scene = Scene(dataset, gaussians) # 创建场景，处理场景初始化，保存和获取相机信息等任务
    gaussians.training_setup(opt) # 设置训练参数
    if checkpoint: # 如果有检查点
        (model_params, first_iter) = torch.load(checkpoint) # 加载检查点的模型参数和迭代次数
        gaussians.restore(model_params, opt) # 恢复模型参数

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0] # 设置背景颜色 如果是白色背景，背景颜色为[1, 1, 1]，否则为[0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 背景颜色转换为张量，数据类型为torch.float32，设备为cuda
    # 创建两个CUDA事件，用于记录迭代开始和结束的时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") # 创建进度条用于显示训练进度
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):   #开始迭代     
        if network_gui.conn == None: # 检查是否连接到GUI服务器，连接则接受消息 
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive() # 接收消息
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"] # 渲染图像
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()) # 将图像转换为字节
                network_gui.send(net_image_bytes, dataset.source_path) # 发送图像
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive): 
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record() # 记录迭代开始时间

        gaussians.update_learning_rate(iteration) # 更新学习率

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree() # 每1000次迭代，增加球谐函数SH的阶数

        # Pick a random Camera 随机选择一个相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy() # 获取训练相机 
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))  # 随机选择一个相机

        # Render 渲染图像，计算损失 L1 和 SSIM
        if (iteration - 1) == debug_from: 
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background # 随机背景

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg) # 渲染图像 
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"] # 获取渲染图像，视空间点张量，可见性过滤器，半径

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # 计算损失 L1 和 SSIM
        loss.backward() # 反向传播

        iter_end.record() # 记录迭代结束时间

        with torch.no_grad(): # 不计算梯度 记录损失的指数移动平均值，更新进度条，保存模型参数，记录训练报告    torch.no_grad()作用：PyTorch将不会追踪Tensor的梯度信息，这样就避免了不必要的内存消耗和计算开销。
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log # 记录损失的指数移动平均值
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"}) # 更新进度条
                progress_bar.update(10) # 更新进度条
            if iteration == opt.iterations: # 如果迭代次数等于opt.iterations
                progress_bar.close() # 关闭进度条

            # Log and save 
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background)) # 记录训练报告
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification 在迭代次数小于opt.densify_until_iter时，记录最大半径，添加稠密化统计信息
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning 
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter]) # 记录最大半径
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) # 添加稠密化统计信息

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0: # 在迭代次数大于opt.densify_from_iter且迭代次数能被opt.densification_interval整除时
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None # 如果迭代次数大于opt.opacity_reset_interval，size_threshold为20，否则为None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold) # 稠密化和修剪
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter): # 在迭代次数能被opt.opacity_reset_interval整除或者是白色背景且迭代次数等于opt.densify_from_iter时
                    gaussians.reset_opacity() # 重置不透明度

            # Optimizer step 优化器步骤
            if iteration < opt.iterations: # 如果迭代次数小于opt.iterations
                gaussians.optimizer.step() # 优化器步骤
                gaussians.optimizer.zero_grad(set_to_none = True) # 优化器梯度清零

            if (iteration in checkpoint_iterations): # 如果迭代次数在checkpoint_iterations中
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):   
    '''
    记录训练过程中的损失和评估指标，并将其可视化到 TensorBoard 中
    ''' 
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser 创建一个参数解析器对象
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG) 初始化系统状态
    safe_state(args.quiet)

    # Start GUI server, configure and run training # 启动GUI服务器，配置并运行训练
    network_gui.init(args.ip, args.port) # 初始化GUI服务器 使用arg.ip和arg.port作为参数
    torch.autograd.set_detect_anomaly(args.detect_anomaly) # 设置PyTorch是否检测梯度计算中异常
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # 输入参数：模型参数，优化参数，管道参数，测试迭代次数，保存迭代次数，检查点迭代次数，开始检查点，调试开始迭代次数

    # All done
    print("\nTraining complete.")
