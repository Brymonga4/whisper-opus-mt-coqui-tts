class Utilities:

    def read_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
        
    def iso_to_lang(iso_code):
        languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'pl': 'Polish',
            'tr': 'Turkish',
            'ru': 'Russian',
            'nl': 'Dutch',
            'cs': 'Czech',
            'ar': 'Arabic',
            'zh-cn': 'Chinese (Simplified)',
            'hu': 'Hungarian',
            'ko': 'Korean',
            'ja': 'Japanese',
            'hi': 'Hindi'
        }
        return languages.get(iso_code, "Language not supported")   
    
    def supported_lang(iso_code):
        languages = {
            'en': 'English',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'ar': 'Arabic'
        }
        return iso_code in languages
