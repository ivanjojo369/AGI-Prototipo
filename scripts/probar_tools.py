from tools.registry import auto_discover, call_tool, is_tool_allowed
import pprint

def main():
    auto_discover(force=True)
    assert is_tool_allowed('web.fetch'), "web.fetch no está permitido por la allowlist"
    out = call_tool('web.fetch', url='https://example.com', max_chars=200)
    pprint.pp(out)

if __name__ == "__main__":
    main()
