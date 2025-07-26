import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import hashlib

__all__=['AESCipher']


class AESCipher:
    def __init__(self, secret_key):
        # 确保密钥是16/24/32字节长
        if len(secret_key) not in [16, 24, 32]:
            raise ValueError("secret_key must be 16 or 24 or 32")
        else:
            self.key = secret_key.encode('utf-8') if isinstance(secret_key, str) else secret_key

    def str_en_base64(self,text,encoding='utf-8'):
        return self.byte_en_base64(text.encode(encoding))
    def byte_en_base64(self, data):
        """加密数据，返回Base64编码的IV+密文"""
        if not data:
            return None

        try:
            # 生成随机IV (16字节)
            iv = get_random_bytes(16)

            # 创建AES-CBC加密器
            cipher = AES.new(self.key, AES.MODE_CBC, iv)

            # 加密数据 (PKCS7填充)
            padded_data = pad(data, AES.block_size)
            ciphertext = cipher.encrypt(padded_data)

            # 组合IV和密文，然后Base64编码
            combined = iv + ciphertext
            return base64.b64encode(combined).decode('utf-8')

        except Exception as e:
            print(f"加密失败: {e}")
            return None


    def base64_de_str(self, encrypted_base64_str,encoding='utf-8'):
        return self.base64_de_byte(encrypted_base64_str).decode(encoding)
    def base64_de_byte(self, encrypted_base64_str):
        """解密Base64编码的IV+密文数据"""
        if not encrypted_base64_str:
            return None

        try:
            # Base64解码
            combined = base64.b64decode(encrypted_base64_str)

            # 提取IV (前16字节)
            iv = combined[:16]
            ciphertext = combined[16:]

            # 创建AES-CBC解密器
            cipher = AES.new(self.key, AES.MODE_CBC, iv)

            # 解密并去除填充
            decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size)
            return decrypted_data

        except ValueError as e:
            print(f"解密失败: 可能是无效的填充 - {e}")
            return None
        except Exception as e:
            print(f"解密失败: {e}")
            return None
