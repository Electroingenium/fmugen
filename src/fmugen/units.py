"""SI base-unit exponents of common unit strings, for <BaseUnit> in modelDescription.xml.

`base_unit("kW.h")` -> {"kg": "1", "m": "2", "s": "-2", "factor": "3600000.0"}; None when the unit isn't
recognised (the unit is then declared by name only). Units are written as in Modelica
and FMI ("m/s2", "W/(m2.K)", "kg.m-3"), with "*", "^" and spaces also accepted.
"""
import math
import re

#: SI base units in the order of the <BaseUnit> attributes (FMI adds rad).
BASE = ("kg", "m", "s", "A", "K", "mol", "cd", "rad")

#: symbol -> (exponents of BASE, factor, offset); value in SI = factor * value + offset.
UNITS = {
    "m": ({"m": 1}, 1.0, 0.0), "s": ({"s": 1}, 1.0, 0.0), "g": ({"kg": 1}, 1e-3, 0.0),
    "A": ({"A": 1}, 1.0, 0.0), "K": ({"K": 1}, 1.0, 0.0), "mol": ({"mol": 1}, 1.0, 0.0),
    "cd": ({"cd": 1}, 1.0, 0.0), "rad": ({"rad": 1}, 1.0, 0.0),
    "degC": ({"K": 1}, 1.0, 273.15), "degF": ({"K": 1}, 5 / 9, 273.15 - 32 * 5 / 9),
    "deg": ({"rad": 1}, math.pi / 180, 0.0),
    "N": ({"kg": 1, "m": 1, "s": -2}, 1.0, 0.0), "Pa": ({"kg": 1, "m": -1, "s": -2}, 1.0, 0.0),
    "J": ({"kg": 1, "m": 2, "s": -2}, 1.0, 0.0), "W": ({"kg": 1, "m": 2, "s": -3}, 1.0, 0.0),
    "V": ({"kg": 1, "m": 2, "s": -3, "A": -1}, 1.0, 0.0), "C": ({"s": 1, "A": 1}, 1.0, 0.0),
    "Ohm": ({"kg": 1, "m": 2, "s": -3, "A": -2}, 1.0, 0.0), "F": ({"kg": -1, "m": -2, "s": 4, "A": 2}, 1.0, 0.0),
    "H": ({"kg": 1, "m": 2, "s": -2, "A": -2}, 1.0, 0.0), "S": ({"kg": -1, "m": -2, "s": 3, "A": 2}, 1.0, 0.0),
    "T": ({"kg": 1, "s": -2, "A": -1}, 1.0, 0.0), "Wb": ({"kg": 1, "m": 2, "s": -2, "A": -1}, 1.0, 0.0),
    "Hz": ({"s": -1}, 1.0, 0.0), "l": ({"m": 3}, 1e-3, 0.0), "L": ({"m": 3}, 1e-3, 0.0),
    "min": ({"s": 1}, 60.0, 0.0), "h": ({"s": 1}, 3600.0, 0.0), "d": ({"s": 1}, 86400.0, 0.0),
    "bar": ({"kg": 1, "m": -1, "s": -2}, 1e5, 0.0), "Wh": ({"kg": 1, "m": 2, "s": -3 + 1}, 3600.0, 0.0),
    "rpm": ({"rad": 1, "s": -1}, 2 * math.pi / 60, 0.0),
}
UNITS["Ω"] = UNITS["Ohm"]
UNITS["°C"] = UNITS["degC"]
UNITS["°"] = UNITS["deg"]
#: Decimal prefixes a unit symbol may carry (``u`` and ``µ`` are both micro).
PREFIXES = {"G": 1e9, "M": 1e6, "k": 1e3, "c": 1e-2, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9}
#: Units that take no prefix (an offset, or a prefix that would read as another unit).
NO_PREFIX = {"degC", "degF", "deg", "°C", "°", "min", "h", "d", "rpm", "K"}

#: Written for other units as often as for these: "C" for Celsius (coulomb), "F" for Fahrenheit
#: (farad), "S" for seconds (siemens), "T" for temperature (tesla). On their own they get no
#: BaseUnit; with a prefix ("uF", "mS") they are unambiguous.
AMBIGUOUS = {"C", "F", "S", "T"}

#: One factor of a product: a symbol, an optional ``^`` and an optional signed integer power.
_FACTOR = re.compile(r"^([^\d^+-]+)\^?([+-]?\d+)?$")


def _symbol(text):
    """The exponents, factor and offset of one symbol, possibly with a prefix: 'km', 'kW', 'mbar'.

    :param text: the symbol.
    :type text: str
    :return: ``(exponents, factor, offset)``, or None when unknown or ambiguous.
    :rtype: tuple or None
    """
    if text in AMBIGUOUS:
        return None
    if text in UNITS:
        return UNITS[text]
    prefix, rest = text[:1], text[1:]
    if prefix in PREFIXES and rest in UNITS and rest not in NO_PREFIX:
        exponents, factor, offset = UNITS[rest]
        return exponents, PREFIXES[prefix] * factor, offset
    return None


def _product(text):
    """Exponents and factor of a product such as 'kg.m2', 'kg*m^2' or 'kg m2' (no division).

    :param text: the product.
    :type text: str
    :return: ``(exponents, factor, offset)``, or None when a factor isn't recognised or an offset
        unit (degC) is combined with others.
    :rtype: tuple or None
    """
    tokens = [t for t in re.split(r"[.*\s·]+", text.strip()) if t and t != "1"]
    exponents, factor, offset = {}, 1.0, 0.0
    for token in tokens:
        match = _FACTOR.match(token)
        found = match and _symbol(match.group(1))
        if not found:
            return None
        power = int(match.group(2) or 1)
        unit_exponents, unit_factor, unit_offset = found
        if unit_offset:
            if len(tokens) > 1 or power != 1:
                return None   # degC only on its own: a temperature difference has no offset
            offset = unit_offset
        for key, value in unit_exponents.items():
            exponents[key] = exponents.get(key, 0) + value * power
        factor *= unit_factor ** power
    return exponents, factor, offset


def parse(unit):
    """The exponents, factor and offset of a unit string, or None if it isn't recognised.

    Accepts one ``/`` with an optional parenthesised denominator ("W/(m2.K)").

    :param unit: the unit string.
    :type unit: str
    :return: ``(exponents without zeros, factor, offset)``, or None.
    :rtype: tuple or None
    """
    text = unit.strip()
    numerator, slash, denominator = text.partition("/")
    denominator = denominator.strip()
    if denominator.startswith("(") and denominator.endswith(")"):
        denominator = denominator[1:-1]
    if "/" in denominator or "(" in numerator + denominator or ")" in numerator + denominator:
        return None
    top = _product(numerator)
    if top is None:
        return None
    exponents, factor, offset = top
    if slash:
        bottom = _product(denominator)
        if bottom is None or bottom[2] or offset:
            return None
        for key, value in bottom[0].items():
            exponents[key] = exponents.get(key, 0) - value
        factor /= bottom[1]
    return {k: v for k, v in exponents.items() if v}, factor, offset


def base_unit(unit):
    """The attributes of <BaseUnit> for a unit string, or None if it isn't recognised.

    :param unit: the unit string.
    :type unit: str
    :return: base-unit exponents as strings, plus ``factor`` / ``offset`` when not 1 / 0.
    :rtype: dict[str, str] or None
    """
    parsed = parse(unit)
    if parsed is None:
        return None
    exponents, factor, offset = parsed
    attrs = {k: str(exponents[k]) for k in BASE if exponents.get(k)}
    if not math.isclose(factor, 1.0, rel_tol=1e-12):
        attrs["factor"] = repr(_round(factor))
    if offset:
        attrs["offset"] = repr(_round(offset))
    return attrs


def display_unit(unit, display):
    """The factor and offset of <DisplayUnit name=display> for a variable in ``unit``.

    FMI: display value = factor * value + offset.

    :param unit: the variable's unit.
    :type unit: str
    :param display: the display unit.
    :type display: str
    :return: ``(factor, offset)``, or None when the two units aren't recognised or don't measure
        the same quantity.
    :rtype: tuple[float, float] or None
    """
    a, b = parse(unit), parse(display)
    if a is None or b is None or a[0] != b[0]:
        return None
    return _round(a[1] / b[1]), _round((a[2] - b[2]) / b[1])


def _round(x):
    """A float without floating-point noise: 3.6, not 3.5999999999999996.

    :param x: the value.
    :type x: float
    :return: ``x`` rounded to 15 significant digits.
    :rtype: float
    """
    return float(f"{x:.15g}")
