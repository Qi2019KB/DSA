# -*- coding: utf-8 -*-
import GLOB as glob
import time
import datetime
import os


class Logger:
    _default_levelConfigs = [
        {'level': 'L2', 'threshold': 90, 'filename': 'log_L2.log'},
        {'level': 'L1', 'threshold': 100, 'filename': 'log_L1.log'}
    ]

    def __init__(self, experiment, consoleLevel, levelConfigs=None):
        self.levelConfigs = self._default_levelConfigs if levelConfigs is None else levelConfigs
        self.loggers = [self._logger(experiment, levelParams) for levelParams in self.levelConfigs]
        self.consoleLevel = consoleLevel

    def print(self, level, content, start=None, end=None):
        content = self._format_level(level, self._format_time(content, start, end))
        if self._check_console(level): print(content)
        for logger in self._get_enable_logger(level):
            loggerObj = logger['logger']
            loggerObj.write(content+'\n')
            loggerObj.flush()
        time.sleep(0.000000001)

    def _logger(self, experiment, levelParams):
        pathname = '{}/{}/logs/{}'.format(glob.expr, experiment, levelParams['filename'])
        folderPath = os.path.split(pathname)[0]
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        logger = open(pathname, 'a+')
        return {'level': levelParams['level'], 'threshold': levelParams['threshold'], 'logger': logger}

    def _format_time(self, content, start=None, end=None):
        now = end if end is not None else datetime.datetime.now()
        interval = '-' if start is None else self._interval_format(seconds=(now - start).seconds)
        return '{} ({}): {}'.format(now.strftime('%m-%d %H:%M'), interval, content)

    def _format_level(self, level, content):
        return '[{}] {}'.format(level, content)

    def _get_enable_logger(self, level):
        loggerObjArray = []
        for logger in self.loggers:
            if logger['threshold'] <= self._get_threshold(level):
                loggerObjArray.append(logger)
        return loggerObjArray

    def _check_console(self, level):
        return self._get_threshold(level) >= self._get_threshold(self.consoleLevel)

    def _get_threshold(self, level):
        return [levelParams for levelParams in self.levelConfigs if levelParams['level'] == level][0]['threshold']

    def _interval_format(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return '%02d:%02d:%02d' % (h, m, s)

