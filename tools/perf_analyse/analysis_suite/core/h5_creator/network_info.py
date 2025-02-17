"""
    get information of network from .json file
"""

__all__ = (
    "NetworkInfo",
)

import os
import logging
import json
import time

class NetworkInfo:
    def __init__(self):
        self.validators = {
            "case_source": self.NetworkConfigFieldValidatorCaseSource,
            "framework": self.NetworkConfigFieldValidatorFramework,
            "is_complete_network": self.NetworkConfigFieldValidatorIsCompleteNetwork,
            "network_name": self.NetworkConfigFieldValidatorNetworkName,
            "batchsize": self.NetworkConfigFieldValidatorBatchsize,
            "precision_mode": self.NetworkConfigFieldValidatorPrecisionMode,
            "card_num": self.NetworkConfigFieldValidatorCardNum,
            "project_version": self.NetworkConfigFieldValidatorProjectVersion,
            "mluops_version": self.NetworkConfigFieldValidatorMluopsVersion,
            "mlu_platform": self.NetworkConfigFieldValidatorMluPlatform,
            "additional_info": self.NetworkConfigFieldValidatorAdditionalInfo,
            "network_property": self.NetworkConfigFieldValidatorNetworkProperty,
            "gen_date": self.NetworkConfigFieldValidatorGenDate,
        }
        self.network_config = {}
        self.case_source_config = (0, 1, 2, 3, 4)
        self.framework_config = ("pt1.6", "pt1.9", "pt1.13", "pt2.1", "pt2.2", "pt2.3", "pt2.4", "tf1.15", "tf2.5", "mm", "customized")
        self.default_network_name = "mluopsbenchmark-all-cloud-operator"
        self.precision_mode_config = ("amp","tf32","bf16","fp16","fp32","qint8_mixed_float16","qint8_mixed_float32","qint8","force_float16","force_float32")
        self.project_version_config = ("CTR_V","INFERENCE_V")
        self.mluops_version_config = ("mluops_v")
        self.default_mlu_platform = ["MLU370", "MLU590", "MLU580B"]

    def analyse_json_config(self, json_file):
        self.json_file_ = json_file
        json_config = self.read_and_check_network_config_json()
        for field, validator in self.validators.items():
            validator(field,json_config)
        return self.network_config

    def read_and_check_network_config_json(self):
        assert os.path.exists(self.json_file_), "{} not exist.".format(self.json_file_)
        with open(self.json_file_) as f:
            json_config = json.load(f)
        assert set(json_config.keys()).issubset(self.validators.keys()), "{} has undefined parameters.".format(self.json_file_)
        return json_config

    def NetworkConfigFieldValidatorCaseSource(self, key, json_config):
        try:
            self.network_config[key] = int(json_config[key])
        except KeyError:
            print("{} is necessary.".format(key))
        except(ValueError,TypeError):
            print("{} must be int type.".format(key))
        else:
            assert self.network_config[key] in self.case_source_config, "{} should be in {}.".format(key,self.case_source_config)

    def NetworkConfigFieldValidatorFramework(self, key, json_config):
        try:
            framework = json_config.get(key,"")
            framework = "" if framework is None else framework
            if self.network_config["case_source"] != 1:
                assert framework != "", "{} is not necessary only when case_source by op owner.".format(key)
                assert framework in self.framework_config, "{} should be in {}.".format(key,'/'.join(self.framework_config))
            else:
                assert framework in self.framework_config or framework == "", "{} should be in {}.".format(key,'/'.join(self.framework_config))
        except TypeError:
            print("{} should be str type.".format(key))
        else:
            self.network_config[key] = framework

    def NetworkConfigFieldValidatorIsCompleteNetwork(self, key, json_config):
        try:
            is_complete_network = json_config[key]
            assert isinstance(is_complete_network,bool), "{} should be bool type.".format(key)
        except:
            print("{} is necessary.".format(key))
        else:
            self.network_config[key] = is_complete_network

    def NetworkConfigFieldValidatorNetworkName(self, key, json_config):
        try:
            if self.network_config["case_source"] == 1:
                network_name = self.default_network_name
            else:
                network_name = json_config[key]
            assert isinstance(network_name,str), "{} should be str type.".format(key)
        except:
            print("{} is  necessary.".format(key))
        else:
            self.network_config[key] = network_name

    def NetworkConfigFieldValidatorBatchsize(self, key, json_config):
        try:
            batchsize = json_config.get(key,0)
            batchsize = 0 if batchsize is None else int(batchsize)
        except(ValueError,TypeError):
            print("{} must be int type.".format(key))
        else:
            self.network_config[key] = batchsize

    def NetworkConfigFieldValidatorCardNum(self, key, json_config):
        try:
            cardnum = json_config.get(key,1)
            cardnum = 1 if cardnum is None else int(cardnum)
        except(ValueError,TypeError):
            print("{} must be int type.".format(key))
        else:
            self.network_config[key] = cardnum

    def NetworkConfigFieldValidatorPrecisionMode(self, key, json_config):
        try:
            precision_mode = json_config.get(key,"")
            precision_mode = "" if precision_mode is None else precision_mode
            assert precision_mode in self.precision_mode_config or precision_mode == "", "{} should be in {}.".format(key,'/'.join(self.precision_mode_config))
        except KeyError:
            print("{} is necessary.".format(key))
        except TypeError:
            print("{} should be str type.".format(key))
        else:
            self.network_config[key] = precision_mode

    def NetworkConfigFieldValidatorProjectVersion(self, key, json_config):
        try:
            project_version = json_config.get(key,"")
            project_version = "" if project_version is None else project_version.upper()
            assert project_version.startswith(self.project_version_config) or project_version == "", "{} should be {}.".format(key,' or '.join([tmp+'x.y.z' for tmp in self.project_version_config]))
        except AttributeError:
            print("{} should be str type.".format(key))
        else:
            self.network_config[key] = project_version

    def NetworkConfigFieldValidatorMluopsVersion(self, key, json_config):
        try:
            mluops_version = json_config[key].lower()
            assert mluops_version.startswith(self.mluops_version_config), "{} should be {}.".format(key,' or '.join([tmp+'x.y.z' for tmp in self.mluops_version_config]))
        except KeyError:
            print("{} is necessary.".format(key))
        except AttributeError:
            print("{} should be str type.".format(key))
        else:
            self.network_config[key] = mluops_version

    def NetworkConfigFieldValidatorMluPlatform(self, key, json_config):
        try:
            mlu_platform = json_config.get(key,self.default_mlu_platform)
            mlu_platform = self.default_mlu_platform if mlu_platform is None else mlu_platform
            if isinstance(mlu_platform,str):
                mlu_platform=mlu_platform.split()
            assert set(mlu_platform).issubset(set(self.default_mlu_platform)), "{} should be in {}".format(key,self.default_mlu_platform)
        except:
            print("{} should be list type.".format(key))
        else:
            self.network_config[key] = (" ").join(mlu_platform)

    def NetworkConfigFieldValidatorAdditionalInfo(self, key, json_config):
        try:
            additional_info = json_config.get(key,"")
            additional_info = "" if additional_info is None else str(additional_info)
        except:
            print("{} should be str type.".format(key))
        else:
            self.network_config[key] = additional_info

    def NetworkConfigFieldValidatorNetworkProperty(self, key, json_config):
        network_property = json_config.get(key,dict())
        network_property = str(dict()) if network_property is None else json.dumps(network_property)
        self.network_config[key] = network_property

    def NetworkConfigFieldValidatorGenDate(self, key, json_config):
        try:
            gen_date=json_config[key]
            gen_date_array = time.strptime(gen_date, "%Y-%m-%d")
            gen_date_out = time.strftime("%Y-%m-%d %H:%M:%S",gen_date_array)
        except KeyError:
            print("{} is necessary.".format(key))
        except TypeError:
            print("{} should be str type.".format(key))
        except ValueError:
            print("{} should be %Y-%m-%d like 2023-11-11.".format(key))
        else:
            self.network_config[key] = gen_date_out
