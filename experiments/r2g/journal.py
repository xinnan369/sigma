"""Transactionally persist immutable, individually hashed image results."""
import hashlib
import json
import sqlite3

from experiments.r1g.partition import require


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def checksum(payload):
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class Journal:
    def __init__(self, path, identity):
        self.db = sqlite3.connect(path, timeout=0)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY, payload TEXT, sha TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, payload TEXT, sha TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS failures (id TEXT PRIMARY KEY, payload TEXT, sha TEXT)')
        payload = encode(identity)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO identity VALUES (1, ?, ?)', (payload, checksum(payload)))
        require(self.db.execute('SELECT payload, sha FROM identity').fetchall() == [(payload, checksum(payload))],
                'Journal identity differs')
        self.identity = identity

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def read(self, *, failures=False):
        table = 'failures' if failures else 'images'
        rows = {}
        for image, payload, sha in self.db.execute(f'SELECT id, payload, sha FROM {table}'):
            require(checksum(payload) == sha, f'Journal row digest differs for {image}')
            value = json.loads(payload)
            require(value['id'] == image and image in self.identity['ids']
                    and value['identity_sha256'] == checksum(encode(self.identity)), 'Journal row identity differs')
            rows[image] = value
        return rows

    def save(self, image, value, *, failure=False):
        require(image in self.identity['ids'], 'Unexpected image')
        row = dict(value, id=image, identity_sha256=checksum(encode(self.identity)))
        payload = encode(row)
        statement = ('INSERT INTO failures VALUES (?, ?, ?)' if failure
                     else 'INSERT INTO images VALUES (?, ?, ?)')
        with self.db:
            self.db.execute(statement, (image, payload, checksum(payload)))

