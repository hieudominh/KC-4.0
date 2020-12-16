# -*- coding: utf-8 -*-
"""
Translation module

We can translate text using this module
"""

import requests

from google_translate.urls import TRANSLATE
from google_translate.utils import format_json
from google_translate.constants import DEFAULT_USER_AGENT, LANGUAGES, SPECIAL_CASES, LANGCODES
from google_translate.gtoken import TokenAcquirer
from google_translate.utils import build_params
# from similarity.core.models.language_detected import LanguageDetected


class Translator(object):
    """
    Google Translate API implementation class

    :param user_agent: The User-Agent header to send when making requests
    :type user_agent: str

    :param proxies: Proxies configuration
                    Dictionary mapping protocol or protocol and host to the URL of the proxy
                    For example ``{'http': 'foo.bar:3128', 'http://host.name': 'foo.bar:4012'}``
    :type proxies: dict

    :param timeout: Definition of timeout for Requests library.
                    Default value: 30 (seconds)
    :type timeout: number

    """

    def __init__(self, user_agent=DEFAULT_USER_AGENT, proxies=None, timeout=30):
        self.session = requests.Session()
        if proxies is not None:
            self.session.proxies = proxies
        self.session.headers.update({
            'User-Agent': user_agent,
        })

        self.service_url = 'translate.google.com'
        self.token_acquirer = TokenAcquirer(session=self.session, host=self.service_url)
        self.timeout = timeout

    def _translate(self, text, dest, src, override):
        token = self.token_acquirer.do(text)
        params = build_params(query=text, src=src, dest=dest, token=token, override=override)
        url = TRANSLATE.format(host=self.service_url)
        res = self.session.get(url=url, params=params)


        data = format_json(res.text)
        return data

    def _parse_extra_data(self, data):
        response_parts_name_mapping = {
            0: 'translation',
            1: 'all-translations',
            2: 'original-language',
            5: 'possible-translations',
            6: 'confidence',
            7: 'possible-mistakes',
            8: 'language',
            11: 'synonyms',
            12: 'definitions',
            13: 'examples',
            14: 'see-also',
        }

        extra = {}

        for index, category in response_parts_name_mapping.items():
            extra[category] = data[index] if (index < len(data) and data[index]) else None

        return extra

    def translate(self, text, dest="en", src="auto", **kwargs):
        """
        Translate text from source language to destination lanaguage

        :param text: The source text(s) to be translated. Batch translation is supported via sequence input.
        :type text: str; list

        :param dest: The language to translate the source text into.
                     The value should be one of the language codes listed in :const:`google_trans.LANGUAGES`
        :type dest: str

        :param src: The language of the source text.
                    The value should be one of the language codes listed in :const:`google_trans.LANGUAGES`
                    If a language is not specified, the system will attempt to identify the source language
                    automatically.
        :type src: str

        :param kwargs:
        :type kwargs:

        :return: Translated
        :rtype:
        """

        dest = dest.lower().split('_', 1)[0]
        src = src.lower().split('_', 1)[0]

        # Validate source language
        if src != 'auto' and src not in LANGUAGES:
            if src in SPECIAL_CASES:
                src = SPECIAL_CASES[src]
            elif src in LANGCODES:
                src = LANGCODES[src]
            else:
                raise ValueError('Invalid source language')

        # Validate destination language
        if dest not in LANGUAGES:
            if dest in SPECIAL_CASES:
                dest = SPECIAL_CASES[dest]
            elif dest in LANGCODES:
                dest = LANGCODES[dest]
            else:
                raise ValueError('Invalid destination language')

        # If bulk translation:
        if isinstance(text, list):
            result = []
            for item in text:
                translated = self.translate(item, dest=dest, src=src, **kwargs)
                result.append(translated)
            return result

        # First, detect language
        data = self._translate(text, 'en', 'auto', kwargs)
        detected_language = ''
        detected_confidence = 0.0
        try:
            detected_language = ''.join(data[8][0])
            detected_confidence = data[8][-2][0]
        except Exception:
            pass

        # Set origin text
        origin = text

        # Then, we will translate origin text
        if dest != 'en':
            # Only re-translate if destination language differences 'en'
            data = self._translate(text, dest, src, kwargs)

        # This code will be updated when the format is changed.
        translated = ''.join([d[0] if d[0] else '' for d in data[0]])

        # Extra data
        extra_data = self._parse_extra_data(data)

        # Actual source language that will be recognized by Google Translator when the
        # src passed is equal to auto.
        try:
            src = data[2]
        except Exception:  # pragma: nocover
            pass

        # Put final values to result object
        # result = LanguageDetected(
        #     src=src, dest=dest, origin=origin, text=translated,
        #     lang=detected_language, confidence=detected_confidence, extra_data=extra_data
        # )

        return translated

    def detect(self, text, **kwargs):
        """
        Detect language of the input text

        :param text: The source text(s) whose language you want to identify.
                     Batch detection is supported via sequence input.
        :type text: str; list

        :param kwargs:
        :type kwargs:

        :return: Detected
        :rtype:
        """

        if isinstance(text, list):
            result = []
            for item in text:
                lang = self.detect(item)
                result.append(lang)
            return result

        data = self._translate(text, 'en', 'auto', kwargs)

        # Actual source language that will be recognized by Google Translator when the
        # src passed is equal to auto.
        src = ''
        confidence = 0.0
        try:
            src = ''.join(data[8][0])
            confidence = data[8][-2][0]
        except Exception:  # pragma: nocover
            pass

        return src, confidence
