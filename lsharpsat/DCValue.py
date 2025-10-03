class DCValue:
    DC = None  # Special value representing a don't-care

    def __init__(self, value=DC):
        self.value = value

    @property
    def is_dc(self):
        return self.value is DCValue.DC

    def __repr__(self):
        if self.value is DCValue.DC:
            return "DC"
        return repr(self.value)

    def __eq__(self, other):
        if self.value is DCValue.DC:
            return True

        if isinstance(other, DCValue):
            if other.value is DCValue.DC:
                return True
            return self.value == other.value

        return self.value == other

    def __bool__(self):
        if self.value is DCValue.DC:
            return False
        return bool(self.value)

    def __hash__(self):
        return hash(self.value)


def common_dc_value(*values: DCValue | bool) -> DCValue | bool:
    if not values:
        return DCValue.DC

    # Find the first non-DC value
    value = values[0]
    for v in values:
        if not (isinstance(v, DCValue) and v.is_dc):
            value = v
            break

    for v in values[0:]:
        if value != v:
            raise ValueError("Values do are not all equal or DC.")

    return value
