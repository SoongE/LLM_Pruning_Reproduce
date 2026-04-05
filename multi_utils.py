import logging


class Style:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    # Foreground Colors
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'


class CustomLogFormatter(logging.Formatter):
    def format(self, record):
        # 1. Format timestamp
        timestamp = self.formatTime(record, "%H:%M:%S")

        # 2. Get custom fields (defaults provided if not present)
        icon = getattr(record, 'icon', "•")
        color = getattr(record, 'color', Style.RESET)
        message = record.getMessage()

        # 3. Construct the final log string
        return f"{Style.BOLD}[{timestamp}]{Style.RESET} {color}{icon} {message}{Style.RESET}"


class SpecificLevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


def setup_logger():
    logger = logging.getLogger("GPUScheduler")
    logger.setLevel(logging.INFO)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Apply our custom formatter
    ch.setFormatter(CustomLogFormatter())

    # Filter specific level
    # ch.addFilter(SpecificLevelFilter(logging.DEBUG))

    # Prevent adding multiple handlers if function is called multiple times
    if not logger.handlers:
        logger.addHandler(ch)

    return logger


def make_layer_combo(start, end, interval):
    layers = []
    for i in range(start, end):
        if i + interval >= end: break
        layers.append(f'{i}-{i + interval}')

    return layers


def make_layer_combo_range(start, end, interval_range):
    layers = []
    for i in range(interval_range):
        layers = layers + make_layer_combo(start, end, i)
    layers = sorted(layers)
    return layers


if __name__ == '__main__':
    l = make_layer_combo_range(9, 32, 22)
    print(len(l))
