import argparse

import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_yaml/config.yaml')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num-epochs', type=int, default=10000)
    parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()

    stages = utils.eval_config(config=utils.load_yaml(args.config))

    if args.mode == 'train':
        trainer = stages['trainer']
        trainer.train(
            num_epochs=args.num_epochs,
            num_gpus=args.num_gpus,
            resume_path=args.resume_path,
            checkpoint_path=args.checkpoint_path,
        )
    elif args.mode == 'test':
        evaluator = stages['evaluator']
        evaluator.eval(
            num_gpus=args.num_gpus,
            checkpoint_path=args.checkpoint_path,
            save_dir=args.save_dir
        )
