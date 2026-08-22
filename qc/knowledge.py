from __future__ import annotations

import argparse
from pathlib import Path

from qc.knowledge_build import KnowledgeBuildService


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5 immutable knowledge builds")
    parser.add_argument("command", choices=["build", "publish", "rollback", "pointer"])
    parser.add_argument("--source", default="knowledge")
    parser.add_argument("--knowledge-version")
    parser.add_argument("--actor", default="cli")
    args = parser.parse_args(argv)
    service = KnowledgeBuildService(Path(args.source))
    if args.command == "build":
        result = service.build()
        print(result.knowledge_version)
    elif args.command in {"publish", "rollback"}:
        if not args.knowledge_version:
            parser.error("--knowledge-version is required")
        print(service.publish(args.knowledge_version, actor=args.actor)["knowledgeVersion"])
    else:
        print(service.current() or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

