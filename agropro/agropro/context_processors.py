def user_info(request):
    if request.user.is_authenticated:
        username = request.user.username
        trimmed_username = username[:username.rfind('-')]
        utype = username[username.rfind('-') + 1:]
        return {
            'username': trimmed_username,
            'utype': utype
        }
    return {}
