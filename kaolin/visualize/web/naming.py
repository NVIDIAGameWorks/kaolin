import secrets


def _sanitize_id_name(name):
    import re
    name = re.sub(r'[^\w\-]', '-', name)
    # Remove consecutive hyphens
    name = re.sub(r'-+', '-', name)
    # Remove leading/trailing hyphens
    name = name.strip('-')
    return name

def generate_random_string(length=4):
    # Use only consonants + digits to minimize chance of forming words
    safe_chars = 'bcdfghjkmnpqrstvwxz'
    return ''.join(secrets.choice(safe_chars) for _ in range(length))


class UniqueIdGenerator:
    _instance = None

    @staticmethod
    def singleton():
        if UniqueIdGenerator._instance is None:
            UniqueIdGenerator._instance = UniqueIdGenerator()
        return UniqueIdGenerator._instance

    def __init__(self, prefix=None):
        self.__ids = set()
        self.prefix = prefix if prefix is not None else ''

    def reset_ids(self, new_ids=None):
        if new_ids is not None:
            self.__ids = set(new_ids)
        else:
            self.__ids = set()

    def get_unique_id_local(self, name=None, prefix=None):
        if prefix is None:
            prefix = self.prefix

        if name is None:
            name = ''

        if name is None or len(name) == 0:
            name = ''
        else:
            name = _sanitize_id_name(name)

        if len(name) == 0:
            name = generate_random_string(3)

        if len(prefix) > 0:
            name = f'{prefix}-{name}'

        if name not in self.__ids:
            self.__ids.add(name)
            return name
        else:
            # append numbers to name, until unique id is found
            counter = 1
            while True:
                candidate = f"{name}_{counter}"
                if candidate not in self.__ids:
                    self.__ids.add(candidate)
                    return candidate
                counter += 1

    @staticmethod
    def get_unique_id(name='', prefix='kaolin'):
        return UniqueIdGenerator.singleton().get_unique_id_local(name, prefix=prefix)


class SequenceWithUniqueGlobalIds:
    """
    A sequence of items with unique IDs, meant to be created one by one, without ability to
    remove item from the sequence. Each item may have a locally unique ID (e.g. "my_cat"),
    and will also be assigned a globally unique id (e.g. "lastname-cat13"), constructed with
    an optional `uid_prefix` and the provided identifier. Each item may be indexed by its
    locally unique ID, global ID or its index in the list using the [] operator. The same
    operator can also be used to mutate items in the sequence, but they cannot be removed.
    """
    def __init__(self, uid_prefix=''):
        self.uid_prefix = uid_prefix
        self._items = []
        self._uid_to_item = {}  # unique ID to item index
        self._id_to_uid = {}  # input ID to unique ID

    def add(self, identifier, item, name_hint=None):
        if identifier is not None:
            if identifier in self._id_to_uid:
                raise RuntimeError(f'Identifier {identifier} already used: {list(self._id_to_uid.keys())}')
            elif identifier in self._uid_to_item:
                raise RuntimeError(f'Identifier {identifier} already used as a unique id: {list(self._uid_to_item.keys())}')

        uid = UniqueIdGenerator.get_unique_id(identifier or name_hint, prefix=self.uid_prefix)
        if identifier is not None:
            self._id_to_uid[identifier] = uid

        self._uid_to_item[uid] = len(self._items)
        self._items.append(item)
        return uid

    def _get_idx(self, key):
        """Convert key (int index, identifier, or unique identifier) to list index.

        Args:
            key: Can be an int (list index), an identifier (original ID passed to add),
                 or a unique identifier (UID returned by add).

        Returns:
            The index in _items.

        Raises:
            KeyError: If string key is not found as identifier or unique identifier.
            IndexError: If int index is out of range.
            TypeError: If key is not int or str.
        """
        if isinstance(key, int):
            idx = key
            if idx < 0 or idx >= len(self._items):
                raise IndexError(f'Index {idx} out of range for list of length {len(self._items)}')
        elif isinstance(key, str):
            if key in self._uid_to_item:
                uid = key
            elif key in self._id_to_uid:
                uid = self._id_to_uid[key]
            else:
                raise KeyError(f'Key "{key}" not found as identifier or unique identifier')
            idx = self._uid_to_item[uid]
        else:
            raise TypeError(f'Key must be int or str, got {type(key).__name__}')
        return idx

    def __getitem__(self, key):
        """Access items by identifier, unique identifier, or int index."""
        return self._items[self._get_idx(key)]

    def __setitem__(self, key, value):
        """Set items by identifier, unique identifier, or int index."""
        self._items[self._get_idx(key)] = value

    @property
    def items(self):
        """Return a copy of all items."""
        return list(self._items)

    @property
    def ids(self):
        """Return a copy of all unique identifiers."""
        return list(self._id_to_uid.keys())

    @property
    def unique_ids(self):
        """Return a copy of all unique identifiers."""
        return list(self._uid_to_item.keys())

    @property
    def unique_ids_to_items(self):
        """Return a copy of all unique identifiers to items."""
        return dict(self._uid_to_item)

    def get_unique_ids_or_raise(self, keys):
        return [self.get_unique_id_or_raise(key) for key in keys]

    def get_unique_id_or_raise(self, key):
        unique_id = self.get_unique_id(key)
        if unique_id is None:
            raise ValueError(f'key {key} not found in {self.ids} or {self.unique_ids}')
        return unique_id

    def get_unique_id(self, key):
        """Get the unique identifier for an item by index, identifier, or unique identifier.
        
        Args:
            key: Can be an int (list index), an identifier (original ID passed to add),
                 or a unique identifier (UID returned by add).
        
        Returns:
            The unique identifier (UID) for the item.
        """
        if isinstance(key, int):
            idx = key
            if idx < 0 or idx >= len(self._items):
                raise IndexError(f'Index {idx} out of range for list of length {len(self._items)}')
            # Find UID by index
            for uid, item_idx in self._uid_to_item.items():
                if item_idx == idx:
                    return uid
            return None
        elif isinstance(key, str):
            if key in self._uid_to_item:
                return key  # Already a unique identifier
            elif key in self._id_to_uid:
                return self._id_to_uid[key]
            else:
                return None
        else:
            raise TypeError(f'Key must be int or str, got {type(key).__name__}')

    def __repr__(self):
        lines = []
        for uid, idx in self._uid_to_item.items():
            # Find identifier if it exists
            identifier = next((k for k, v in self._id_to_uid.items() if v == uid), None)
            id_str = f"id='{identifier}', " if identifier else ""
            lines.append(f"  [{idx}] {id_str}uid='{uid}': {self._items[idx]!r}")
        return "\n".join(lines)