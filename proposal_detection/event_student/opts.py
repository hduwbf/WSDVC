import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=8)
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--training_lr',
        type=float,
        default=1e-3)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5)

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=20)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32)
    parser.add_argument(
        '--step_size',
        type=int,
        default=15)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.5)

    # Overall Dataset settings
    parser.add_argument(
        '--anno_path',
        type=str,
        default="../data/event/")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="../data/event/AnetFeatures.hdf5")

    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=500)

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=8)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.75)
    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.5)
    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.8)
    parser.add_argument(
        '--result_file',
        type=str,
        default="./output/result_proposal.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="./output/evaluation_result.jpg")

    args = parser.parse_args()

    return args

