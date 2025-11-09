To train the executor in the paper, run this command:
python main.py --env Pose-v0 --model single-att --workers 6

To train the coordinator in the paper, run this command:
python main.py --env Pose-v1 --model multi-att-shap --workers 6

To evaluate my model, run:
python main.py --env Pose-v1 --render --model multi-att-shap --workers 0 --load-coordinator-dir trainedModel/best_coordinator.pth --load-executor-dir trainedModel/best_executor.pth