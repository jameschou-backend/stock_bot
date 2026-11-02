"""US acquisition only: no signals, POC, Taiwan database access or broker calls."""
def ingest(query, store, client=None, *, max_pages=100):
    if max_pages < 1:
        raise ValueError("Positive page budget required")
    with store.lock():
        store.capacity()
        url, urls, identities, pages = query.url(), set(), set(), []
        for index in range(max_pages):
            url = query.validate_url(url)
            if url in urls:
                raise query.error_type("Pagination cursor loop")
            urls.add(url)
            saved = store.page(query,index)
            if saved:
                receipt, payload = saved
                if receipt["url"] != url:
                    raise ValueError("Cached pagination chain differs from requested chain")
            else:
                if client is None:
                    raise query.error_type("Offline snapshot is incomplete; explicit --fetch and API key required")
                payload = client.page(url)
            table = query.normalize_page(payload)
            # A repeated provider identity is an acquisition conflict, NOT grounds
            # to drop equal timestamp/price trades. Genuine distinct trades remain.
            if query.dataset in ("daily","trades"):
                for row in table.to_pylist():
                    identity = query.row_identity(row)
                    if identity in identities:
                        raise ValueError("Duplicate provider identity across snapshot pages")
                    identities.add(identity)
            next_url = query.next_url(payload)
            if not saved:
                receipt = store.save_page(query,index,url,payload,table)
            elif receipt["rows"] != table.num_rows:
                raise ValueError("Cached row count mismatch")
            pages.append(receipt)
            if not next_url:
                return store.publish(query,pages)
            url = next_url
        raise query.error_type("Page budget exhausted; incomplete snapshot was not published")
