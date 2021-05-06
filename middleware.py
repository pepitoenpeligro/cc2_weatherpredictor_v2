class LoggerMiddleware(object):
    def __init__(self, app):
        self.app = app
    def __call__(self, environ, start_response):
        print ('[V2] Request received from ', environ['REMOTE_ADDR'], ' to resource: ',  environ['RAW_URI'], flush=True)
        return self.app(environ, start_response)