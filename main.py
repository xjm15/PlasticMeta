import logging
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from configs.regression import parser as reg_parser
from datasets import sampler as task_sampler
from models import factory as mf
from experiments.tracker import ExperimentTracker
from models.meta_learner import MetaLearner
from utils import helpers as utils
from utils.Dataset import SequenceDataset

logger = logging.getLogger('experiment')


def main():
    p = reg_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)
    utils.set_seed(args["seed"])

    my_experiment = ExperimentTracker(args["name"], args, "../results/", commit_changes=False,
                                      rank=int(rank / total_seeds),
                                      seed=total_seeds)

    my_experiment.results["all_args"] = all_args

    writer = SummaryWriter(my_experiment.path + "tensorboard")

    base_folder = r'../metadrive-data/meta_task_noO'
    m1 = 3
    n1 = 10
    m2 = 1
    n2 = 15
    all_files1 = []
    all_files2 = []

    folders = [folder for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]

    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        mid_point = len(files) // 2

        if len(files) % 2 == 0:
            all_files1.extend([(folder, file) for file in files[:mid_point]])
            all_files2.extend([(folder, file) for file in files[mid_point:]])
        else:
            all_files1.extend([(folder, file) for file in files[:mid_point + 1]])
            all_files2.extend([(folder, file) for file in files[mid_point + 1:]])

    dataset = SequenceDataset(root_dir=base_folder, transform=None, length=5)

    model_config = mf.ModelFactory.get_model(None, "transformer", input_dimension=259,
                                             output_dimension=2,
                                             width=128, d_ff=512)

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    metalearner = MetaLearner(args, model_config).to(device)

    tmp = filter(lambda x: x.requires_grad, metalearner.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    logger.info('Total trainable tensors: %d', num)

    running_meta_loss = 0
    adaptation_loss = 0
    loss_history = []
    adaptation_loss_history = []
    adaptation_running_loss_history = []
    meta_steps_counter = 0
    LOG_INTERVAL = 2

    for step in range(args["step"]):

        if step % LOG_INTERVAL == 0:
            logger.debug("####\t STEP %d \t####", step)

        meta_steps_counter += 1

        sampler1 = task_sampler.TaskSampler(base_folder, all_files1, dataset, m1, n1)
        sampler2 = task_sampler.TaskSampler(base_folder, all_files2, dataset, m2, n2)

        sampled_files = sampler1.sample()
        x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []

        for folder_name, file, idxs in sampled_files:
            for t, idx in enumerate(idxs):
                feature, label = dataset.get_data_from_file(folder_name, file, idx)
                if feature is None: continue  # Skip if data loading failed
                if t < n1 / 2:
                    x_traj.append(feature)
                    y_traj.append(label)
                else:
                    x_rand_temp.append(feature)
                    y_rand_temp.append(label)

        sampled_files = sampler2.sample()
        for folder_name, file, idxs in sampled_files:
            for t, idx in enumerate(idxs):
                feature, label = dataset.get_data_from_file(folder_name, file, idx)
                if feature is None: continue
                x_rand.append(feature)
                y_rand.append(label)

        try:
            y_rand = torch.stack(y_rand).unsqueeze(0)
            x_rand = torch.stack(x_rand).unsqueeze(0)
            y_rand_temp = torch.stack(y_rand_temp).unsqueeze(0)
            x_rand_temp = torch.stack(x_rand_temp).unsqueeze(0)
            x_traj, y_traj = torch.stack(x_traj).unsqueeze(1), torch.stack(y_traj).unsqueeze(1)
        except RuntimeError:
            logger.warning(f"Skipping step {step} due to empty data lists after sampling.")
            continue

        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)

        if torch.cuda.is_available():
            x_traj_meta, y_traj_meta, x_rand_meta, y_rand_meta = x_traj.to(device), y_traj.to(
                device), x_rand.to(
                device), y_rand.to(
                device)

        meta_loss, features = metalearner(x_traj_meta, y_traj_meta, x_rand_meta, y_rand_meta)

        loss_history.append(meta_loss[-1].detach().cpu().item())

        running_meta_loss = running_meta_loss * 0.97 + 0.03 * meta_loss[-1].detach().cpu()
        running_meta_loss_fixed = running_meta_loss / (1 - (0.97 ** (meta_steps_counter)))
        writer.add_scalar('/metatrain/train/accuracy', meta_loss[-1].detach().cpu(), meta_steps_counter)
        writer.add_scalar('/metatrain/train/runningaccuracy', running_meta_loss_fixed,
                          meta_steps_counter)

        if step % LOG_INTERVAL == 0:
            with torch.no_grad():
                mean, log_std = metalearner.net(x_rand_meta[0], vars=None)
                outputs = torch.tanh(mean)
                current_adaptation_loss = F.mse_loss(outputs, y_rand_meta[0])

                adaptation_loss_history.append(current_adaptation_loss.detach().item())
                adaptation_loss = adaptation_loss * 0.97 + current_adaptation_loss.detach().cpu().item() * 0.03
                adaptation_loss_fixed = adaptation_loss / (1 - (0.97 ** (step + 1)))
                adaptation_running_loss_history.append(adaptation_loss_fixed)

                logger.info(f"Step = {step}, Adaptation loss = {current_adaptation_loss}")
                logger.info(f"Step = {step}, Running adaptation loss = {adaptation_loss_fixed}")
                writer.add_scalar('/learn/test/adaptation_loss', current_adaptation_loss, step)

        if step > 200000:
            LOG_INTERVAL = 40
        elif step > 100000:
            LOG_INTERVAL = 20
        elif step > 50000:
            LOG_INTERVAL = 10
        else:
            LOG_INTERVAL = 2

        if (step) % (LOG_INTERVAL * 1000) == 0:
            if not args["no_save"]:
                torch.save(metalearner.net, my_experiment.path + "net" + str(step) + ".model")
            my_experiment.add_result("Meta loss", loss_history)
            my_experiment.add_result("Adaptation loss", adaptation_loss_history)
            my_experiment.add_result("Running adaption loss", adaptation_running_loss_history)
            my_experiment.store_json()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()