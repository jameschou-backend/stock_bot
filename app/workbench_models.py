"""Cash-account plan and fill ledger. Market data remains in existing MySQL tables."""
from sqlalchemy import Column, String, Numeric, DateTime, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.mysql import DATETIME
from datetime import datetime
from app.models import Base


class WorkbenchAccount(Base):
    __tablename__ = 'workbench_accounts'
    account_id = Column(String(32), primary_key=True)
    initial_cash = Column(Numeric(18, 2), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class WorkbenchPlan(Base):
    __tablename__ = 'workbench_plans'
    plan_id = Column(String(32), primary_key=True)
    account_id = Column(String(32), ForeignKey('workbench_accounts.account_id'), nullable=False, index=True)
    stock_id = Column(String(4), nullable=False)
    entry_price = Column(Numeric(18, 4), nullable=False)
    stop_price = Column(Numeric(18, 4), nullable=False)
    qty = Column(Integer, nullable=False)
    filled_qty = Column(Integer, nullable=False, default=0)
    status = Column(String(16), nullable=False, default='open')
    reason = Column(String(500), nullable=False, default='')
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class WorkbenchFill(Base):
    __tablename__ = 'workbench_fills'
    __table_args__ = (UniqueConstraint('account_id', 'sequence_no'),)
    fill_id = Column(String(32), primary_key=True)
    sequence_no = Column(Integer, nullable=False)
    account_id = Column(String(32), ForeignKey('workbench_accounts.account_id'), nullable=False, index=True)
    plan_id = Column(String(32), ForeignKey('workbench_plans.plan_id'), nullable=True)
    stock_id = Column(String(4), nullable=False)
    side = Column(String(4), nullable=False)
    qty = Column(Integer, nullable=False)
    price = Column(Numeric(18, 4), nullable=False)
    fee = Column(Numeric(18, 2), nullable=False)
    tax = Column(Numeric(18, 2), nullable=False)
    executed_at = Column(DateTime().with_variant(DATETIME(fsp=6), 'mysql'), nullable=False)
    created_at = Column(DateTime().with_variant(DATETIME(fsp=6), 'mysql'), nullable=False, default=datetime.utcnow)


TABLES = [WorkbenchAccount.__table__, WorkbenchPlan.__table__, WorkbenchFill.__table__]
