# Python Security

> Extends common/security.md with Python-specific content.

## Secret Management

```python
import os

api_key = os.environ["API_KEY"]  # Raises KeyError if missing
```

- NEVER hardcode secrets in source code
- Use environment variables or `.env` files (excluded from git)
- Validate required secrets at startup

## Security Scanning

Use **bandit** for static security analysis:
```bash
bandit -r src/
```

## Common Vulnerabilities

### Command Injection

```python
# BAD: Shell injection risk
os.system(f"cat {user_input}")

# GOOD: Use subprocess with list args
subprocess.run(["cat", user_input], check=True)
```

### Path Traversal

```python
# BAD: User controls path
path = base_dir / user_input

# GOOD: Validate path stays within base
path = (base_dir / user_input).resolve()
if not path.is_relative_to(base_dir.resolve()):
    raise ValueError("Path traversal detected")
```

### Unsafe Deserialization

```python
# BAD: pickle with untrusted data
import pickle
data = pickle.loads(untrusted_bytes)

# GOOD: Use safe formats
import json
data = json.loads(untrusted_string)
```

### Weak Crypto

- Never use MD5/SHA1 for security purposes
- Use `hashlib.sha256` or better for hashing
- Use `bcrypt` or `argon2` for password hashing

## Before ANY Commit

- [ ] No hardcoded secrets (API keys, passwords, tokens)
- [ ] All user inputs validated
- [ ] Error messages don't leak sensitive data
