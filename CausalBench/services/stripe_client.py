"""
Stripe API Client for CausalBench

Uses stripe-python for real Stripe API interactions.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .base_client import BaseServiceClient, APIResponse, register_service

logger = logging.getLogger(__name__)

try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    logger.warning("stripe not installed. Run: pip install stripe")


class StripeClient(BaseServiceClient):
    """
    Stripe API client for payment and customer operations.
    """

    SERVICE_NAME = "stripe"
    TRUST_LEVEL = "sensitive"  # Financial data is sensitive

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs
    ):
        api_key = api_key or os.environ.get("STRIPE_API_KEY")
        super().__init__(api_key=api_key, **kwargs)

    def _initialize_client(self) -> bool:
        if not STRIPE_AVAILABLE:
            logger.error("stripe not available")
            return False
        if not self.api_key:
            logger.error("STRIPE_API_KEY not set")
            return False
        try:
            stripe.api_key = self.api_key
            # Test connection - list one customer
            stripe.Customer.list(limit=1)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Stripe client: {e}")
            return False

    def list_items(self, limit: int = 10, item_type: str = "customers", **kwargs) -> APIResponse:
        """List customers, charges, or subscriptions."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "customers":
                result = stripe.Customer.list(limit=limit)
                items = [
                    {
                        "id": c.id,
                        "email": c.email,
                        "name": c.name,
                        "created": c.created,
                        "currency": c.currency,
                        "balance": c.balance,
                        "delinquent": c.delinquent
                    }
                    for c in result.data
                ]

            elif item_type == "charges":
                result = stripe.Charge.list(limit=limit)
                items = [
                    {
                        "id": c.id,
                        "amount": c.amount,
                        "currency": c.currency,
                        "status": c.status,
                        "customer": c.customer,
                        "created": c.created,
                        "description": c.description
                    }
                    for c in result.data
                ]

            elif item_type == "subscriptions":
                result = stripe.Subscription.list(limit=limit)
                items = [
                    {
                        "id": s.id,
                        "customer": s.customer,
                        "status": s.status,
                        "current_period_start": s.current_period_start,
                        "current_period_end": s.current_period_end,
                        "plan": s.plan.id if s.plan else None
                    }
                    for s in result.data
                ]

            elif item_type == "invoices":
                result = stripe.Invoice.list(limit=limit)
                items = [
                    {
                        "id": i.id,
                        "customer": i.customer,
                        "amount_due": i.amount_due,
                        "amount_paid": i.amount_paid,
                        "status": i.status,
                        "created": i.created
                    }
                    for i in result.data
                ]

            elif item_type == "payment_methods":
                customer = kwargs.get("customer")
                if not customer:
                    return APIResponse(success=False, error="customer parameter required")
                result = stripe.PaymentMethod.list(customer=customer, type="card", limit=limit)
                items = [
                    {
                        "id": pm.id,
                        "type": pm.type,
                        "card_brand": pm.card.brand if pm.card else None,
                        "card_last4": pm.card.last4 if pm.card else None,
                        "created": pm.created
                    }
                    for pm in result.data
                ]

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data={"items": items, "count": len(items)})

        except stripe.error.StripeError as e:
            return APIResponse(success=False, error=str(e), status_code=e.http_status)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def get_item(self, item_id: str, item_type: str = "customer", **kwargs) -> APIResponse:
        """Get a specific customer, charge, or subscription."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "customer":
                c = stripe.Customer.retrieve(item_id)
                data = {
                    "id": c.id,
                    "email": c.email,
                    "name": c.name,
                    "phone": c.phone,
                    "address": dict(c.address) if c.address else None,
                    "created": c.created,
                    "balance": c.balance,
                    "metadata": dict(c.metadata)
                }

            elif item_type == "charge":
                c = stripe.Charge.retrieve(item_id)
                data = {
                    "id": c.id,
                    "amount": c.amount,
                    "currency": c.currency,
                    "status": c.status,
                    "customer": c.customer,
                    "receipt_url": c.receipt_url,
                    "description": c.description
                }

            elif item_type == "subscription":
                s = stripe.Subscription.retrieve(item_id)
                data = {
                    "id": s.id,
                    "customer": s.customer,
                    "status": s.status,
                    "cancel_at_period_end": s.cancel_at_period_end,
                    "current_period_start": s.current_period_start,
                    "current_period_end": s.current_period_end
                }

            elif item_type == "invoice":
                i = stripe.Invoice.retrieve(item_id)
                data = {
                    "id": i.id,
                    "customer": i.customer,
                    "amount_due": i.amount_due,
                    "amount_paid": i.amount_paid,
                    "status": i.status,
                    "hosted_invoice_url": i.hosted_invoice_url
                }

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

            return APIResponse(success=True, data=data)

        except stripe.error.StripeError as e:
            return APIResponse(success=False, error=str(e), status_code=e.http_status)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def create_item(self, data: Dict[str, Any], item_type: str = "customer", **kwargs) -> APIResponse:
        """Create a customer, charge, or refund."""
        if not self.ensure_initialized():
            return APIResponse(success=False, error="Client not initialized")

        if self.simulate_writes:
            logger.info(f"[SIMULATED] Would create {item_type}: {data}")
            return APIResponse(
                success=True,
                data={"simulated": True, "item_type": item_type, "would_create": data}
            )

        self.rate_limiter.wait_if_needed()

        try:
            if item_type == "customer":
                c = stripe.Customer.create(
                    email=data.get("email"),
                    name=data.get("name"),
                    metadata=data.get("metadata", {})
                )
                return APIResponse(success=True, data={"id": c.id, "email": c.email})

            elif item_type == "charge":
                c = stripe.Charge.create(
                    amount=data.get("amount"),
                    currency=data.get("currency", "usd"),
                    customer=data.get("customer"),
                    description=data.get("description")
                )
                return APIResponse(success=True, data={"id": c.id, "status": c.status})

            else:
                return APIResponse(success=False, error=f"Unknown item_type: {item_type}")

        except stripe.error.StripeError as e:
            return APIResponse(success=False, error=str(e), status_code=e.http_status)
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    def create_refund(self, charge_id: str, amount: Optional[int] = None) -> APIResponse:
        """Create a refund (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would refund charge {charge_id}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "create_refund",
                "charge": charge_id,
                "amount": amount
            }
        )

    def create_payout(self, amount: int, currency: str = "usd") -> APIResponse:
        """Create a payout (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would create payout of {amount} {currency}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "create_payout",
                "amount": amount,
                "currency": currency
            }
        )

    def update_customer_email(self, customer_id: str, email: str) -> APIResponse:
        """Update customer email (SENSITIVE - always simulated)."""
        logger.warning(f"[SIMULATED] Would update {customer_id} email to {email}")
        return APIResponse(
            success=True,
            data={
                "simulated": True,
                "action": "update_customer_email",
                "customer": customer_id,
                "new_email": email
            }
        )


# Register the client
register_service("stripe", StripeClient)
