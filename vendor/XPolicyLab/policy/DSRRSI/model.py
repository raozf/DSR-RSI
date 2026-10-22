"""Explicitly selected DSR execution protocol."""
import os
if os.environ.get('DSR_AGENT_STRATEGY') == 'physicalrsi':
    from dsr.system1.physicalrsi_runtime import Model
else:
    from XPolicyLab.policy.OpenDM.model import Model
