from src.browser.Browser import Browser
from src.sentence.Sentence import Sentence
from src.translator.Translatable import Translatable
import time

DEFAULT_SOURCE_LANGID = 'auto'
SOURCE_BOX_XPATH = '//textarea[@class="er8xn"]'
TARGET_BOX_XPATH = '//span[@jsname="W297wb"]'
prev_text = ''


class GoogleTranslate(Translatable):

    def __init__(self, browser: Browser):
        super().__init__()

        self.source_langid = DEFAULT_SOURCE_LANGID

        self.browser = browser
        self.url = None

    def update(self):
        if self.source_langid is None or self.target_langid is None:
            raise ValueError('The source lang id and the target lang id must be provided')
        self.url = 'https://translate.google.com/?sl=' + self.source_langid + '&tl=' + self.target_langid + '&op=translate'
        self.browser.load_web_page(self.url)

    def set_source_langid(self, langid: str):
        if langid is None or langid == '':
            raise ValueError('The source lang id must be provided')
        self.source_langid = langid
        if self.target_langid is not None:
            self.update()

    def set_target_langid(self, langid: str):
        if langid is None or langid == '':
            raise ValueError('The target lang id must be provided')
        self.target_langid = langid
        if self.source_langid is not None:
            self.update()

    def translate(self, text: str) -> str:
        if self.url is None:
            raise ValueError('Please provide source language id and target language id!')

        global prev_text

        def find_trans_box(driver):
            element = driver.find_element_by_xpath(TARGET_BOX_XPATH)
            # print(element.text)
            if element and element.text == prev_text:
                # print("bug")
                return False
            if element and element.text != prev_text:
                return element
            else:
                return False

        def find_src_box(driver):
            element = driver.find_element_by_xpath(SOURCE_BOX_XPATH)
            if element:
                return element
            else:
                return False

        src_text = self.browser.run_and_wait(50, find_src_box)
        src_text.clear()
        src_text.send_keys(text)
        time.sleep(1)
        # WebDriverWait(driver, 50).until(EC.presence_of_element_located((By.XPATH, '//div[@class="text-wrap tlid-copy-target"]')))
        dst_text = self.browser.run_and_wait(50, find_trans_box)
        time.sleep(1)
        # dst_text = driver.find_element_by_xpath('//div[@class="text-wrap tlid-copy-target"]')
        prev_text = dst_text.text
        return dst_text.text
