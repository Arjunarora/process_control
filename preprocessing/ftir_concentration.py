from tensorflow import math


def get_saturation_concentration(temperature: float, substance: str) -> float:
    """
    Calculates the saturation concentration of a substance at given temperature in water.
    :param temperature: System temperature in °C
    :param substance: Substance name in lowercase, e.g. "adipic_acid" or "kh2po4"
    :return: The saturation concentration of a given substance in water
    """
    if substance in ["adipic_acid", "aa", "adipin"]:
        concentration_saturation = 13.505 * math.exp(0.0418 * temperature)
    elif substance in ["potassium_dihydrogen_phosphate", "kh2po4", "pdp", "kdp"]:
        temperature += 273.15
        concentration_saturation = (4.6479 * 10 ** -5 * temperature ** 2 - 0.022596 * temperature + 2.8535) * 1000
    else:
        raise NotImplementedError(f"feature_engineering.get_saturation_concentration: Substance {substance} is not implemented yet.")
    return concentration_saturation


def get_supersaturation(concentration: float, temperature: float, substance: str) -> float:
    """
    Calculates the oversaturation of a substance in water at given temperature
    :param concentration: Concentration of the substance in the system in g/kg_H2O
    :param temperature: System temperature in °C
    :param substance: Substance name in lowercase, e.g. "adipic_acid" or "kh2po4"
    :return:
    """
    concentration_saturation = get_saturation_concentration(temperature, substance)
    supersaturation = concentration / concentration_saturation
    return supersaturation
