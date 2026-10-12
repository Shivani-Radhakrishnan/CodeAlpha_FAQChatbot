### Answers from the policy documents (98 test questions: 68 answerable, 20 not covered, 10 off-topic)

Thresholds chosen on the dev split: min_relevance 0.25, min_support 0.5.

| System | overall | answerable correct | answerable refused | unanswerable declined | offtopic declined | citation correct | retrieval recall@3 |
|---|---|---|---|---|---|---|---|
| FAQ matcher (existing) | 44.9% | 47.1% | 13.2% | 15.0% | 90.0% | - | - |
| LLM, no documents | 14.3% | 16.2% | 23.5% | 15.0% | 0.0% | - | - |
| RAG, no grounding check | 83.7% | 88.2% | 1.5% | 60.0% | 100.0% | 86.6% | 98.5% |
| RAG with grounding check | 82.7% | 85.3% | 2.9% | 65.0% | 100.0% | 93.9% | 98.5% |

How the final system's answers came about: grounded 69, low relevance 16, model declined 7, partly grounded 4, unsupported answer 2

Final system, every wrong test answer:

| Question | Kind | Answer | Cited |
|---|---|---|---|
| can I send back a personalised engraved bracelet | answerable | Yes, you can send back a personalised engraved bracelet if it arrives damaged or faulty within 7 days of delivery. | rewards_discounts#gift-cards |
| are final sale items returnable | answerable | Yes, final sale items are returnable within 30 days of delivery. | returns#return-window |
| my tracking link shows nothing yet | answerable | It seems that your tracking information has not yet appeared after dispatch. | shipping#tracking-your-order |
| how long is the warranty | answerable | The warranty is 3 years. | warranty#extended-protection |
| my headphones stopped charging after 6 months is that covered | answerable | I couldn't find that in our policies, so I don't want to guess. Type 'human' to ask our support team. |  |
| can I add botbuddy care a month after buying | answerable | Yes, you can add BotBuddy Care a month after buying it. | warranty#extended-protection |
| can I buy without making an account | answerable | You can buy without making an account, but you can't track orders, start returns, or earn rewards points. | account_privacy#creating-an-account |
| can I stop getting order emails | answerable | Yes, you can stop receiving order emails by going to Cookie Settings. | account_privacy#emails-and-cookies |
| do gift cards expire | answerable | Yes, gift cards expire after 1 year from the date of purchase. | rewards_discounts#gift-cards |
| how quickly do you answer emails | answerable | I couldn't find that in our policies, so I don't want to guess. Type 'human' to ask our support team. |  |
| do you price match your own past prices | unanswerable | Yes, if you find the same item cheaper at another major retailer within 14 days of your purchase, we will refund the difference. You need to | rewards_discounts#price-matching |
| can I change the name on a gift card | unanswerable | Yes, you can change the name on a gift card if you send us the purchase confirmation. | rewards_discounts#gift-cards |
| what is the warranty on gift cards | unanswerable | The warranty on gift cards covers defects in materials and workmanship that appear with normal use, such as a device that stops charging or  | rewards_discounts#gift-cards |
| do you recycle old electronics | unanswerable | However, we offer BotBuddy Care, which extends the warranty by 2 more years and adds cover for one accidental-damage repair. | warranty#extended-protection |
| can I get an invoice with my company tax number | unanswerable | Yes, you can get an invoice with your company tax number. We keep order and invoice records for 7 years because tax law requires it, but the | account_privacy#deleting-your-account |
| can I schedule a delivery time slot | unanswerable | You can change the delivery address from your account settings until the order is shipped. After dispatch, the address can no longer be chan | shipping#changing-the-delivery-address |
| do you repair products bought somewhere else | unanswerable | Yes, if you buy a product from another store and it arrives faulty, you can make a warranty claim through BotBuddy. We will repair or replac | warranty#extended-protection, warranty#making-a-warranty-claim |
