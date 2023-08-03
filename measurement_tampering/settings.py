from diamonds.apply_model_to_data import DiamondsSetting
from func_correct.apply_model_to_data import FuncCorrectSetting
from measurement_tampering.train_utils import Setting
from money_making.apply_model_to_data import MoneyMakingNewSetting
from money_making_easy.apply_model_to_data import MoneyMakingSetting
from text_properties.apply_model_to_data import TextPropertiesSetting

SETTINGS = {
    "func_correct": FuncCorrectSetting,
    "diamonds": DiamondsSetting,
    "text_properties": TextPropertiesSetting,
    "money_making": MoneyMakingSetting,
    "money_making_new": MoneyMakingNewSetting,
}


def get_setting(setting_name: str) -> Setting:
    return SETTINGS[setting_name]()
