import os 
import sys

class AutoMLException(Exception):

    def __init__(self, error_mesasge, error_details:sys):
        self.error_message = error_mesasge
        _,_, exc_tb = error_details.exc_info()

        self.line_number = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
    
    def __str__(self):
        return f"Error occurred in script: {self.file_name} at line number: {self.line_number} with message: {str(self.error_message)}"
    