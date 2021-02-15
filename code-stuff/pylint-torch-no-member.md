# [Problem: Pylint E1101 Module 'torch' has no 'from_numpy' member](https://github.com/pytorch/pytorch/issues/701)

## 1. Solution for VSCode

For those using vscode, add to user settings:


```json
"python.linting.pylintArgs": [
"--errors-only",
"--generated-members=numpy.* ,torch.* ,cv2.* , cv.*"
]
```

## 2. FYI:

Depending on your platform, the user settings file is located here:

- Windows: `%APPDATA%\Code\User\settings.json`
- macOS: `$HOME/Library/Application Support/Code/User/settings.json`
- Linux: `$HOME/.config/Code/User/settings.json`