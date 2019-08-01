import codecs
import configparser
import global_variables as gl


class ReadConfig:

    def __init__(self, config_path):

        self.config_path = config_path

        fd = open(config_path)

        data = fd.read()

        if data[:3] == codecs.BOM_UTF8:

            data = data[3:]

            file = codecs.open(config_path, "w")

            file.write(data)

            file.close()

        fd.close()

        self.cf = configparser.ConfigParser()

        self.cf.read(config_path)

    def get_section(self):

        sections = self.cf.sections()

        return sections

    def get_items(self, section):

        items = self.cf.items(section)

        return items

    def set_paras(self, section, option, value):

        self.cf.set(section, option, value)

        with open(self.config_path, 'w+') as f:

            self.cf.write(f)


def para_init(config_path):

    gl._init()

    config = ReadConfig(config_path)

    sections = config.get_section()

    for i in range(len(sections)):

        items = list(config.get_items(sections[i]))

        for k in range(len(items)):

            gl.set_value(items[k][0], items[k][1])

