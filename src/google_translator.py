from googletrans import Translator

class GoogleTranslator:
    def __init__(self) -> None:
        self.translator = Translator()

    def isEnglishOrKorean(self, input_s):
        k_count = 0
        e_count = 0
        for c in input_s:
            if ord('가') <= ord(c) <= ord('힣'):
                k_count+=1
            elif ord('a') <= ord(c.lower()) <= ord('z'):
                e_count+=1
        return 0 if k_count>e_count else 1

    def translate(self, text):
        #encText = urllib.parse.quote(text)
        res = ""
        if text != "":
            if self.isEnglishOrKorean(text) == 0:
                result = self.translator.translate(text, dest="en")
            else:
                result = self.translator.translate(text, dest="ko")
            res = result.text
        return res




