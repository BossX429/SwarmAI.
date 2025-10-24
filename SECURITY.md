# Security Policy

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report vulnerabilities via GitHub Security Advisories:

1. Navigate to the **Security** tab
2. Click **"Report a vulnerability"**
3. Provide detailed information about the vulnerability

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**:
  - Critical: Within 7 days
  - High: Within 30 days
  - Medium: Within 90 days

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Features

- **Agent Isolation**: Agents run in separate processes with resource limits
- **Pheromone Security**: Pheromone trails decay to prevent poisoning attacks
- **Task Validation**: All tasks validated before swarm assignment  
- **Consensus Verification**: Multi-agent voting prevents manipulation
- **Dependency Scanning**: Dependabot monitors for vulnerable dependencies
- **Code Scanning**: CodeQL analyzes code for security issues

## Best Practices

- Never commit API keys or credentials
- Use environment variables for sensitive configuration
- Keep dependencies up to date
- Enable MFA on your GitHub account
- Validate all task inputs
- Monitor swarm behavior for anomalies
- Limit agent capabilities to minimum required

---

*Last Updated: October 24, 2025*
