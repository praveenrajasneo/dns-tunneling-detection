"""
parser.py
---------
Extracts DNS queries from PCAP files or live network traffic using Scapy.
"""

from scapy.all import rdpcap, sniff, DNS, DNSQR, IP
import pandas as pd


def parse_pcap(filepath: str) -> pd.DataFrame:
    """Parse a PCAP file and extract DNS query records."""
    packets = rdpcap(filepath)
    records = []

    for pkt in packets:
        record = _extract_dns_record(pkt)
        if record:
            records.append(record)

    return pd.DataFrame(records)


def sniff_live(interface: str = "eth0", count: int = 100) -> pd.DataFrame:
    """Capture live DNS packets from a network interface."""
    records = []

    def process_packet(pkt):
        record = _extract_dns_record(pkt)
        if record:
            records.append(record)

    sniff(iface=interface, filter="udp port 53", prn=process_packet, count=count)
    return pd.DataFrame(records)


def _extract_dns_record(pkt) -> dict:
    """Extract relevant fields from a single DNS packet."""
    if not (pkt.haslayer(DNS) and pkt.haslayer(DNSQR)):
        return None

    dns = pkt[DNS]
    query = pkt[DNSQR]

    return {
        "src_ip": pkt[IP].src if pkt.haslayer(IP) else None,
        "dst_ip": pkt[IP].dst if pkt.haslayer(IP) else None,
        "query_name": query.qname.decode("utf-8", errors="ignore").rstrip("."),
        "query_type": query.qtype,
        "response_code": dns.rcode,
        "answer_count": dns.ancount,
        "timestamp": float(pkt.time),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parser.py <path_to_pcap>")
    else:
        df = parse_pcap(sys.argv[1])
        print(f"Extracted {len(df)} DNS records")
        print(df.head())
