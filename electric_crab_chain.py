"""
Electric Crab Chain Publisher

Standalone on-chain proof publisher for Electric Crab.

Purpose:
- Publish only the prediction_hash on-chain
- Keep full prediction JSON off-chain
- Return transaction hash for verification

Install:
    pip install web3

Environment variables:
    CHAIN_RPC_URL
    CHAIN_PRIVATE_KEY
    CHAIN_ACCOUNT_ADDRESS
    CHAIN_CONTRACT_ADDRESS

PowerShell example:
    $env:CHAIN_RPC_URL="https://your-rpc-url"
    $env:CHAIN_PRIVATE_KEY="0xYOUR_PRIVATE_KEY"
    $env:CHAIN_ACCOUNT_ADDRESS="0xYOUR_WALLET_ADDRESS"
    $env:CHAIN_CONTRACT_ADDRESS="0xDEPLOYED_CONTRACT_ADDRESS"

CLI examples:
    python electric_crab_chain.py publish demo-001 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

    python electric_crab_chain.py publish demo-001 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef ipfs://your-metadata-uri

    python electric_crab_chain.py get demo-001
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional

from web3 import Web3


class ElectricCrabChainPublisher:
    """
    Publishes Electric Crab prediction hashes to an EVM-compatible chain.

    It stores:
    - event_id
    - prediction_hash
    - optional metadata_uri
    - timestamp
    - publisher address

    It does not upload the full prediction result on-chain.
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        account_address: Optional[str] = None,
        contract_address: Optional[str] = None,
    ):
        self.rpc_url = rpc_url or os.getenv("CHAIN_RPC_URL", "")
        self.private_key = private_key or os.getenv("CHAIN_PRIVATE_KEY", "")
        self.account_address = account_address or os.getenv("CHAIN_ACCOUNT_ADDRESS", "")
        self.contract_address = contract_address or os.getenv("CHAIN_CONTRACT_ADDRESS", "")

        self.enabled = all([
            self.rpc_url,
            self.private_key,
            self.account_address,
            self.contract_address,
        ])

        self.web3 = None
        self.contract = None

        if self.enabled:
            self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))

            if not self.web3.is_connected():
                raise RuntimeError("Unable to connect to chain RPC.")

            self.account_address = Web3.to_checksum_address(self.account_address)
            self.contract_address = Web3.to_checksum_address(self.contract_address)

            self.contract = self.web3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi(),
            )

    def publish_prediction_hash(
        self,
        event_id: str,
        prediction_hash: str,
        metadata_uri: str = "",
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "published": False,
                "chain_status": "CHAIN_NOT_CONFIGURED",
                "reason": (
                    "Missing CHAIN_RPC_URL, CHAIN_PRIVATE_KEY, "
                    "CHAIN_ACCOUNT_ADDRESS, or CHAIN_CONTRACT_ADDRESS."
                ),
                "tx_hash": None,
            }

        validation = self.validate_prediction_hash(prediction_hash)

        if not validation["valid"]:
            return {
                "published": False,
                "chain_status": "INVALID_HASH",
                "reason": validation["reason"],
                "tx_hash": None,
            }

        normalized_hash = self.normalize_prediction_hash(prediction_hash)

        nonce = self.web3.eth.get_transaction_count(self.account_address)

        tx = self.contract.functions.recordPrediction(
            str(event_id),
            normalized_hash,
            str(metadata_uri),
        ).build_transaction({
            "from": self.account_address,
            "nonce": nonce,
            "gas": 250000,
            "gasPrice": self.web3.eth.gas_price,
        })

        signed_tx = self.web3.eth.account.sign_transaction(
            tx,
            private_key=self.private_key,
        )

        raw_tx = getattr(signed_tx, "rawTransaction", None)

        if raw_tx is None:
            raw_tx = signed_tx.raw_transaction

        tx_hash = self.web3.eth.send_raw_transaction(raw_tx)

        return {
            "published": True,
            "chain_status": "PENDING",
            "tx_hash": tx_hash.hex(),
            "event_id": event_id,
            "prediction_hash": prediction_hash.lower().replace("0x", ""),
            "metadata_uri": metadata_uri,
            "submitted_at": int(time.time()),
            "contract_address": self.contract_address,
            "publisher": self.account_address,
        }

    def wait_for_receipt(
        self,
        tx_hash: str,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "confirmed": False,
                "reason": "Chain publisher is not configured.",
            }

        receipt = self.web3.eth.wait_for_transaction_receipt(
            tx_hash,
            timeout=timeout,
        )

        return {
            "confirmed": receipt.status == 1,
            "chain_status": "CONFIRMED" if receipt.status == 1 else "FAILED",
            "tx_hash": tx_hash,
            "block_number": receipt.blockNumber,
            "gas_used": receipt.gasUsed,
            "contract_address": self.contract_address,
        }

    def get_prediction(
        self,
        event_id: str,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "found": False,
                "chain_status": "CHAIN_NOT_CONFIGURED",
                "reason": (
                    "Missing CHAIN_RPC_URL, CHAIN_PRIVATE_KEY, "
                    "CHAIN_ACCOUNT_ADDRESS, or CHAIN_CONTRACT_ADDRESS."
                ),
            }

        result = self.contract.functions.getPrediction(str(event_id)).call()

        prediction_hash_bytes = result[0]
        metadata_uri = result[1]
        timestamp = result[2]
        publisher = result[3]

        prediction_hash = prediction_hash_bytes.hex()

        found = timestamp != 0

        return {
            "found": found,
            "event_id": event_id,
            "prediction_hash": prediction_hash,
            "metadata_uri": metadata_uri,
            "timestamp": timestamp,
            "publisher": publisher,
            "contract_address": self.contract_address,
        }

    @staticmethod
    def normalize_prediction_hash(prediction_hash: str) -> bytes:
        value = prediction_hash.lower().strip()

        if value.startswith("0x"):
            value = value[2:]

        return bytes.fromhex(value)

    @staticmethod
    def validate_prediction_hash(prediction_hash: str) -> Dict[str, Any]:
        if not isinstance(prediction_hash, str):
            return {
                "valid": False,
                "reason": "prediction_hash must be a string.",
            }

        value = prediction_hash.lower().strip()

        if value.startswith("0x"):
            value = value[2:]

        if len(value) != 64:
            return {
                "valid": False,
                "reason": "prediction_hash must be a 64-character SHA-256 hex string.",
            }

        try:
            bytes.fromhex(value)
        except ValueError:
            return {
                "valid": False,
                "reason": "prediction_hash contains non-hex characters.",
            }

        return {
            "valid": True,
            "reason": "OK",
        }

    @staticmethod
    def contract_abi():
        return [
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "eventId",
                        "type": "string",
                    },
                    {
                        "internalType": "bytes32",
                        "name": "predictionHash",
                        "type": "bytes32",
                    },
                    {
                        "internalType": "string",
                        "name": "metadataURI",
                        "type": "string",
                    },
                ],
                "name": "recordPrediction",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function",
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "eventId",
                        "type": "string",
                    },
                ],
                "name": "getPrediction",
                "outputs": [
                    {
                        "internalType": "bytes32",
                        "name": "predictionHash",
                        "type": "bytes32",
                    },
                    {
                        "internalType": "string",
                        "name": "metadataURI",
                        "type": "string",
                    },
                    {
                        "internalType": "uint256",
                        "name": "timestamp",
                        "type": "uint256",
                    },
                    {
                        "internalType": "address",
                        "name": "publisher",
                        "type": "address",
                    },
                ],
                "stateMutability": "view",
                "type": "function",
            },
        ]


def print_json(data: Dict[str, Any]):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def print_usage():
    print(
        """
Electric Crab Chain Publisher

Usage:

  Publish prediction hash:
    python electric_crab_chain.py publish <event_id> <prediction_hash> [metadata_uri]

  Publish and wait for confirmation:
    python electric_crab_chain.py publish-wait <event_id> <prediction_hash> [metadata_uri]

  Read prediction by event id:
    python electric_crab_chain.py get <event_id>

Environment variables required:
  CHAIN_RPC_URL
  CHAIN_PRIVATE_KEY
  CHAIN_ACCOUNT_ADDRESS
  CHAIN_CONTRACT_ADDRESS
"""
    )


def main():
    args = sys.argv[1:]

    if not args:
        print_usage()
        return

    command = args[0].lower().strip()

    try:
        publisher = ElectricCrabChainPublisher()
    except Exception as exc:
        print_json({
            "ok": False,
            "error": str(exc),
        })
        return

    if command == "publish":
        if len(args) < 3:
            print_usage()
            return

        event_id = args[1]
        prediction_hash = args[2]
        metadata_uri = args[3] if len(args) >= 4 else ""

        result = publisher.publish_prediction_hash(
            event_id=event_id,
            prediction_hash=prediction_hash,
            metadata_uri=metadata_uri,
        )

        print_json(result)
        return

    if command == "publish-wait":
        if len(args) < 3:
            print_usage()
            return

        event_id = args[1]
        prediction_hash = args[2]
        metadata_uri = args[3] if len(args) >= 4 else ""

        result = publisher.publish_prediction_hash(
            event_id=event_id,
            prediction_hash=prediction_hash,
            metadata_uri=metadata_uri,
        )

        if result.get("published") and result.get("tx_hash"):
            receipt = publisher.wait_for_receipt(result["tx_hash"])
            result["receipt"] = receipt
            result["chain_status"] = receipt.get("chain_status", result.get("chain_status"))

        print_json(result)
        return

    if command == "get":
        if len(args) < 2:
            print_usage()
            return

        event_id = args[1]

        result = publisher.get_prediction(event_id=event_id)

        print_json(result)
        return

    print_usage()


if __name__ == "__main__":
    main()