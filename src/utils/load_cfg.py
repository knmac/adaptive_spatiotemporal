"""Load YAML config files"""
import sys
import os

import yaml

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
import src.utils.logging as logging

logger = logging.get_logger(__name__)


class ConfigLoader:
    @staticmethod
    def _load_yaml_content(fname):
        """Load and check content of a given YAML file name
        Args:
            fname: path to the config file
        Return:
            content: content of the YAML file
        """
        # Try to use absolute path if file not found
        if not os.path.isfile(fname):
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            fname = os.path.join(root, fname)
        assert os.path.isfile(fname), 'Config file not found: {}'.format(fname)

        with open(fname, 'r') as stream:
            try:
                content = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(exc)
                sys.exit(-1)
        return content

    @staticmethod
    def load_model_cfg(fname):
        """Load model config from a given YAML file name
        Parse some configuration parameters if needed.
        Args:
            fname: path to the config file
        Return:
            Name of the model
            The model parameters as a dictionary
        """
        cfg = ConfigLoader._load_yaml_content(fname)
        if cfg['model_params'] is None:
            cfg['model_params'] = {}
        return cfg['model_name'], cfg['model_params']

    @staticmethod
    def load_dataset_cfg(fname):
        """Load dataset config from a given YAML file name.
        Parse some configuration parameters if needed.
        Args:
            fname: path to the config file
        Return:
            Name of the dataset
            The dataset parameters as a dictionary
        """
        cfg = ConfigLoader._load_yaml_content(fname)
        return cfg['dataset_name'], cfg['dataset_params']

    @staticmethod
    def load_train_cfg(fname):
        """Load training config from a given YAML name.
        Parse some configuration parameters if needed.
        Args:
            fname: path to the config file
        Return:
            The training parameters as a dictionary
        """
        cfg = ConfigLoader._load_yaml_content(fname)
        return cfg
