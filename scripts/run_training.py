#!/usr/bin/env python3
"""
Fuz-RL Training Launcher

A Python script for launching Fuz-RL training with GPU specification and
parameter validation.
"""

import argparse
import os
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description='Fuz-RL Training Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Algorithm selection
    parser.add_argument('-a', '--algorithm', required=True,
                       choices=['ppolag', 'cpo', 'cup', 'fuzppolag', 'fuzcpo', 'fuzcup',
                               'cvarppolag', 'cvarcpo', 'cvarcup'],
                       help='Algorithm to use')

    # Environment settings
    parser.add_argument('-e', '--env', required=True,
                       help='Environment name (e.g., CartPole-v1, SafetyPointPush0-v0)')
    parser.add_argument('-t', '--task', default='stab',
                       choices=['stab', 'track'],
                       help='Task type for safe control environments')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('-s', '--seed', type=int, default=42,
                       help='Random seed')

    # Fuzzy-specific parameters
    parser.add_argument('--level', type=int, default=10,
                       help='Fuzzy level for fuzzy algorithms')
    parser.add_argument('--eps', type=float, default=0.1,
                       help='Initial perturbation magnitude')
    parser.add_argument('--train-eps', type=float, default=1.0,
                       help='Training perturbation magnitude')
    parser.add_argument('--disturb-part', choices=['dynamics', 'observation', 'action', 'multi_source'],
                       default='dynamics', help='Part of system to disturb')
    parser.add_argument('--disturb-type', choices=['white_noise', 'impulse', 'periodic'],
                       default='white_noise', help='Type of disturbance')

    # CVaR-specific parameters
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='CVaR confidence level')

    # Additional options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show command without executing')
    parser.add_argument('--log-dir', default='./logs',
                       help='Log directory')

    # SwanLab integration
    parser.add_argument('--use-swanlab', action='store_true',
                       help='Enable SwanLab logging')
    parser.add_argument('--swanlab-project', default='Fuz-RL',
                       help='SwanLab project name')
    parser.add_argument('--swanlab-api-key', default=None,
                       help='SwanLab API key')

    args = parser.parse_args()

    # Validate GPU availability (basic check)
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available on this system")
        elif args.gpu >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu} not available, using GPU 0")
            args.gpu = 0
    except ImportError:
        print("Warning: PyTorch not available for GPU validation")

    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Build command
    cmd = [
        sys.executable, 'scripts/train.py',
        '--algorithm', args.algorithm,
        '--env', args.env,
        '--task', args.task,
        '--epochs', str(args.epochs),
        '--device', f'cuda:{args.gpu}',
        '--seed', str(args.seed),
        '--log_dir', args.log_dir
    ]

    # Add fuzzy-specific parameters
    if args.algorithm.startswith('fuz'):
        cmd.extend([
            '--level', str(args.level),
            '--eps', str(args.eps),
            '--train_eps', str(args.train_eps),
            '--disturb_part', str(args.disturb_part),
            '--disturb_type', str(args.disturb_type)
        ])

    # Add CVaR-specific parameters
    if args.algorithm.startswith('cvar'):
        cmd.extend(['--alpha', str(args.alpha)])

    # Add SwanLab parameters
    if args.use_swanlab:
        cmd.extend([
            '--use_swanlab',
            '--swanlab_project', str(args.swanlab_project)
        ])
        if args.swanlab_api_key:
            cmd.extend(['--swanlab_api_key', str(args.swanlab_api_key)])

    # Print configuration
    print("🚀 Fuz-RL Training Launcher")
    print("=" * 40)
    print(f"Algorithm:     {args.algorithm}")
    print(f"Environment:   {args.env}")
    print(f"Task:          {args.task}")
    print(f"Epochs:        {args.epochs}")
    print(f"GPU:           {args.gpu}")
    print(f"Seed:          {args.seed}")
    if args.algorithm.startswith('fuz'):
        print(f"Fuzzy Level:   {args.level}")
        print(f"EPS:           {args.eps}")
        print(f"Train EPS:     {args.train_eps}")
    print(f"Disturb Part:  {args.disturb_part}")
    print(f"Disturb Type:  {args.disturb_type}")
    print(f"SwanLab:       {'Enabled' if args.use_swanlab else 'Disabled'}")
    if args.use_swanlab:
        print(f"Project:       {args.swanlab_project}")
    if args.algorithm.startswith('cvar'):
        print(f"CVaR Alpha:    {args.alpha}")
    print("=" * 40)

    # Show command
    cmd_str = ' '.join(cmd)
    print(f"Command: {cmd_str}")
    print()

    if args.dry_run:
        print("🔍 Dry run mode - command shown above, not executed")
        return

    # Execute training
    print("Starting training...")
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()
