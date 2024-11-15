import hashlib
import uuid


def get_hash_from_uuid(hash_len: int = 5) -> str:
    # Generate a UUID4 and convert it to a string
    uuid_str = str(uuid.uuid4())

    # Hash the UUID string using SHA-256
    hash_object = hashlib.sha256(uuid_str.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig[:hash_len]
