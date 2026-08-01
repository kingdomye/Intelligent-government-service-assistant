# Security Policy

## Secrets

Do not commit API tokens, encryption keys, model credentials, `.env` files, production user data, real form samples, member photos, student records, or unredacted training data. Copy `.env.example` to `.env` for local configuration.

Keep private datasets under `private/`, `local-data/`, or `training-data/`; all three directories are ignored. Use synthetic fixtures in tests and examples.

The ModelArts credential previously stored in `Code/qgai/datamining/deal_flow_api1.py` must be revoked and replaced. Removing it from the current tree does not remove it from existing clones or Git history.

If the repository is public, rewrite the affected history only after coordinating with all contributors because the operation changes commit IDs and requires force-pushing every affected branch and tag.

## Reporting

Report vulnerabilities privately to the repository maintainers. Do not include active credentials or personal data in a public issue.
