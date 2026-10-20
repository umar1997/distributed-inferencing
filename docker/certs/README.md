# Extra trusted certificates (optional)

Some company networks route all internet traffic through a security filter (for example
**Zscaler**). It opens encrypted (HTTPS) connections, checks them, and signs them again with its
own certificate. Your Mac trusts that certificate (it is in the Keychain), but a fresh Docker
container does not, so `pip install` and model downloads inside Docker fail with
`CERTIFICATE_VERIFY_FAILED`.

Put such a certificate here as a `.crt` file (PEM text format) and the Docker image will trust it.
Certificate checks stay switched on; the image simply trusts one more issuer.

On a Mac with Zscaler:

```bash
security find-certificate -c "Zscaler Root CA" -p /Library/Keychains/System.keychain > docker/certs/zscaler-root-ca.crt
```

`.crt` / `.pem` files in this folder are git-ignored (they belong to your network, not to the
project). On a machine without such a filter, leave the folder empty: nothing changes.
