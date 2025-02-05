# Python Repo Template

This is a template for a Python repository following organizational conventions and best practices.

After creating a repo using this template, configure the `pyproject.toml` file with the appropriate metadata for your project.

Once complete, remove the `template-configuration-needed` file and rebuild your dev container.

## How to clone this template for a new project
This project is not set as GitHub template due to the banner polution it causes for attribution.
Instead of creating new repos using the template function in the GitHub web UI, create an _empty_ repo and clone the
template project as follows (replacing `project-name` accordingly):

```bash
PROJECT_NAME="project-name"
git clone git@github.com:latchfield/template-python.git ${PROJECT_NAME}
cd ${PROJECT_NAME}
git remote rename origin template
git remote add origin git@github.com:latchfield/${PROJECT_NAME}.git
git reset --soft $(git rev-list --max-parents=0 HEAD)
git commit -m "Squashed template"
git push -u origin main
git branch --track template template/main
git config remote.template.push refs/heads/template:refs/heads/main
git config push.default current
```

## How to merge changes from the template into your project
If you want to merge changes from the template into your project, you can do so as follows:

```bash
git fetch template

# Latest
git merge --no-commit template/main

# Specific version tag (e.g.)
git merge --no-commit template/1.0.0
```

# Development
## Testing
### Run Tests
#### Non Integration Tests:
```bash
pytest
```

#### Integration Tests:
```bash
pytest --integration
```

## Dependency Management
### Review major updates
```bash
poetry show --outdated --latest -T
```

### Update Dependencies
```bash
poetry up
```

### Update only lockfile
```bash
poetry update
```

### Audit Security Vulnerabilities
```bash
poetry audit
```

### Verify Dependencies
```bash
deptry src
```

### Check for CVEs
```bash
poetry audit
```

### Check for insecure coding
*Note: Mostly replaced by ruff, but good for reporting*
```bash
bandit -r src
```

### Check for unused dependencies
```bash
deptry src
```

## Misc Commands
### Clear pycache
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

### Show compact git history
```bash
git --no-pager log --oneline --graph --all
```