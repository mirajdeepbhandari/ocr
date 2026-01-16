import ssl
import warnings


def disable_ssl():
    """Fix SSL certificate verification issues on Windows."""
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    ssl._create_default_https_context = ssl._create_unverified_context
    
    import requests
    from functools import partialmethod
    requests.Session.request = partialmethod(requests.Session.request, verify=False)
    requests.request = partialmethod(requests.request, verify=False)
