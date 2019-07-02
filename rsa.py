import random


def exgcd(a, b, x, y):
    if b == 0:
        x = 1
        y = 0
        return a, x, y
    gcd, x, y = exgcd(b, a % b, x, y)
    x2, y2 = x, y
    x = y2
    y = x2 - int(a/b) * y2
    return gcd, x, y


def rsa_code(p, q):
    n = p * q
    o = (p - 1) * (q - 1)
    e = int(random.random() * (o - 2) + 2)
    while coprime(e, o) != 1:
        e = int(random.random() * (o - 2) + 2)
    gcd, x, y = exgcd(e, o, 0, 0)
    return n, e, x


def coprime(a, b):
    if a < b:
        t = a
        a = b
        b = t
    while a % b != 0:
        r = a % b
        a = b
        b = r
    return b


def coding(text, keys):
    return text ** keys[1] % keys[0]



def create_keys():
    prime = input('Input p and q:').split(" ")
    p, q = int(prime[0]), int(prime[1])
    n, e, d = rsa_code(p, q)
    public_key = (n, e)
    private_key = (n, d)
    return public_key, private_key


pub, pri = create_keys()
print("明文为：65")
print("公钥为(%s, %s)" % (pub[0], pub[1]))
print("私钥为(%s, %s)" % (pri[0], pri[1]))
print('----------------')
password = coding(65, pub)
print("RSA加密后:%s" % hex(password))
print("RSA解密后:%s" % coding(password, pri))