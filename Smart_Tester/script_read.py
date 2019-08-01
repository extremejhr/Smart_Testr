import re


class ScriptRead(object):

    def __init__(self, script_path):

        self.script = open(script_path)

    def remove_comments(self):

        script_processed = []

        while True:

            line = self.script.readline()

            line_slim = line.replace(' ', '')

            if len(line) == 0:

                break

            if not ((line_slim == '\n') or (line_slim[0] == '#')):

                script_processed.append(line.replace('\n', ''))

        return script_processed

    def complete_commands(self, script):

        script_processed = []

        script_prev_window = ''

        flag_in_prev = ''

        for i in range(len(script)):

            script_tmp = script[i].replace(' ', '')

            if '@' in script_tmp.lower():

                script_in= script[i].lower()

                flag_in = 'RESET'

                script_processed.append([script_in,  flag_in])

                script_prev_window = script[i].split('=>')[0].lower()

                flag_in_prev = 'RESET'

            elif 'and' in script_tmp.lower():

                script_processed.append([script_prev_window + '=>' + script[i].lower().replace('and', ''), flag_in_prev])

            elif '>>' in script_tmp.lower():

                script_processed.append([script_prev_window + '=>' + script[i].lower().replace('>>', ''), 'REMAIN'])

            else:

                script_processed.append([script[i].lower(), 'SYSTEM'])

        return script_processed

    def script_unzip(self, script):

        pattern = re.compile("[\[](.*?)[\]]")

        script_processed = []

        for i in range(len(script)):

            act_type = script[i][1]

            if act_type in ['RESET', 'REMAIN']:

                script_comma = script[i][0].split('=>')

                child_canvas = re.findall(pattern, script_comma[0])[0]

                if child_canvas in ['top_border', 'ribbon_up', 'ribbon_down', 'quick_access',
                                    'navigator', 'main_window', 'left_border']:

                    mother_canvas = 'main'

                else:

                    mother_canvas = 'dialog'

                act_key = re.findall(pattern, script_comma[1])[0]

                if '->' in script_comma[1]:

                    script_arrow = script_comma[1].split('->')

                    target = re.findall(pattern, script_arrow[1])[0]

                    pattern_anchor = re.compile("[<](.*?)[>]")

                    anchor_all = re.findall(pattern_anchor, target)

                    if len(anchor_all) == 0:

                        anchor = 'invalid'

                    else:

                        anchor_tmp = '<' + anchor_all[0] + '>'

                        target = target.replace(anchor_tmp, '')

                        anchor = anchor_all[0]

                else:

                    target = 'invalid'

                    anchor = 'invalid'

                if '=' in script_comma[1]:

                    script_equal = script_comma[1].split('=')

                    input = re.findall(pattern, script_equal[1])[0]

                else:

                    input = 'invalid'

                script_processed.append([mother_canvas, child_canvas, act_key, target, input, act_type, anchor])

            else:

                act1 = script[i][0].split(' ')[0]

                if act1 == 'wait':

                    input_time = eval(script[i][0].split(' ')[1])

                    script_processed.append(['system', 'invalid', 'wait', 'invalid', input_time, 'invalid', 'invalid'])

                elif act1 == 'finish':

                    script_processed.append(['system', 'invalid', 'finish', 'invalid', 'invalid', 'invalid', 'invalid'])

        return script_processed

    def script_select_def(self, script):

        script_processed = []

        for i in range(len(script)):

            if script[i][2] in ['rselect', 'bselect']:

                print(script[i])

                script_tmp = script[i].copy()

                script_tmp1 = script[i].copy()

                if script[i][2] == 'rselect':

                    script_tmp[2] = 'rmb1'

                elif script[i][2] == 'bselect':

                    script_tmp[2] = 'bmb1'

                else:

                    script_tmp[2] = ''

                    print('Wrong RSELECT and BSELECT !!!!')

                script_tmp[4] = 'invalid'

                script_tmp1[2] = 'mb1'

                script_tmp1[3] = script[i][4]

                script_tmp1[4] = 'invalid'

                script_tmp1[-1] = 'REMAIN'

                script_processed.append(script_tmp)

                script_processed.append(script_tmp1)

            else:

                script_processed.append(script[i])

        return script_processed


def script_process(inp_path):

    script_op = ScriptRead(inp_path)

    pre = script_op.remove_comments()

    full_scirpt = script_op.complete_commands(pre)

    unziped_script = script_op.script_unzip(full_scirpt)

    compiled_script = script_op.script_select_def(unziped_script)

    return compiled_script
