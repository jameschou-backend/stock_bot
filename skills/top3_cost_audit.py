"""Private copies of sealed cash/unit auditors with a fixed stress commission.

Every reconciliation assertion is retained. Only the single hard-coded fee
literal changes, after source-hash and occurrence-count checks. This does not
patch shared imported modules or accept the executor's own costs as proof.
"""
import hashlib
import inspect
from pathlib import Path
from types import FunctionType, SimpleNamespace

from skills.top3_validation import MODES

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = {
    'skills/corporate_account_audit.py': 'b2d5cb374737ac2bdcee82e3f474d3e196dded72dfbf4e5670aa48a9c9e3a00d',
    'skills/face_value_capital.py': '2da5e707338923740f463ba6a72ae9f2cda3f4722d3365fa754da35b5b732f1c',
    'skills/execution_resources.py': 'b644a6ba3aba0acb0ddf6769ad604d8c2508738115ca45fd980b3895a535c8f3',
    'skills/execution_stress.py': 'a88874a9187e1e97fef7417c17d4098915bc0b8548b42ee6b41b6086a472cdf6',
}


def private_cash_audit(function, mode):
    path = Path(inspect.getsourcefile(function)).resolve()
    relative = str(path.relative_to(ROOT))
    if hashlib.sha256(path.read_bytes()).hexdigest() != BINDINGS[relative]:
        raise ValueError('Frozen cash auditor changed')
    code = inspect.getsource(function)
    literal = "raw_gross*Decimal('.001425')"
    if code.count(literal) != 1:
        raise ValueError('Expected exactly one fixed commission audit site')
    code = code.replace(literal, 'raw_gross*Decimal(str(_top3_commission))')
    namespace = dict(function.__globals__, _top3_commission=MODES[mode][0])
    exec(compile(code, str(path), 'exec'), namespace)
    return namespace[function.__name__]


def install(runner, mode):
    from skills.corporate_account_audit import audit_corporate_account
    from skills.execution_stress import audit_stress
    from skills.execution_resources import audit_resources
    from skills.face_value_capital import audit_face_resources
    for name, expected in BINDINGS.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen resource audit changed: '+name)
    corporate = private_cash_audit(audit_corporate_account, mode)
    benchmark = private_cash_audit(audit_stress, mode)
    # Retain the original capital action -> slot-only projection and all checks.
    code = inspect.getsource(audit_face_resources)
    site = 'from skills.corporate_account_audit import audit_corporate_account'
    if code.count(site) != 1:
        raise ValueError('Expected the original corporate audit binding')
    code = code.replace(site, 'audit_corporate_account = _top3_corporate_audit')
    namespace = dict(audit_face_resources.__globals__, _top3_corporate_audit=corporate)
    exec(compile(code, str(ROOT/'skills/face_value_capital.py'), 'exec'), namespace)
    runner.audit_face_resources = namespace['audit_face_resources']
    resources = FunctionType(audit_resources.__code__,
        dict(audit_resources.__globals__, audit_stress=benchmark),
        audit_resources.__name__, audit_resources.__defaults__, audit_resources.__closure__)
    resources.__kwdefaults__ = audit_resources.__kwdefaults__
    # install_private_index has already isolated these shallow namespaces.
    runner.study = SimpleNamespace(**vars(runner.study))
    runner.study.old = SimpleNamespace(**vars(runner.study.old))
    runner.study.old.audit_resources = resources
