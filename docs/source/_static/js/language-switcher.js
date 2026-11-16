/*
 * SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

(() => {
  const STORAGE_KEY = "cutile-docs-lang";
  const hasNativePopover = Object.hasOwn(HTMLElement.prototype, "popover");
  const hasInvokerCommands = "commandForElement" in HTMLButtonElement.prototype;
  const hasAnchorPos =
    typeof CSS !== "undefined" &&
    typeof CSS.supports === "function" &&
    CSS.supports("position-area: block-end");

  function positionMenuFallback(trigger, menu) {
    if (hasAnchorPos) return;
    const rect = trigger.getBoundingClientRect();
    const menuRect = menu.getBoundingClientRect();
    const margin = 6;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let left = rect.right - menuRect.width;
    if (left < 8) left = rect.left;
    left = Math.max(8, Math.min(left, vw - menuRect.width - 8));

    let top = rect.bottom + margin;
    if (top + menuRect.height > vh - 8 && rect.top - menuRect.height - margin >= 8) {
      top = rect.top - menuRect.height - margin;
    }

    menu.style.left = `${Math.round(left)}px`;
    menu.style.top = `${Math.round(top)}px`;
    menu.style.right = "auto";
  }

  function initSwitcher(root) {
    const trigger = root.querySelector(".lang-switcher__trigger");
    const menu = root.querySelector(".lang-switcher__menu");
    if (!trigger || !menu) return;

    const setExpanded = (open) => {
      trigger.setAttribute("aria-expanded", open ? "true" : "false");
    };

    if (hasNativePopover) {
      menu.addEventListener("toggle", (event) => {
        const isOpen = event.newState === "open";
        setExpanded(isOpen);
        if (isOpen) {
          positionMenuFallback(trigger, menu);
        }
      });

      if (!hasInvokerCommands) {
        trigger.addEventListener("click", () => {
          if (menu.matches(":popover-open")) {
            menu.hidePopover();
          } else {
            try {
              menu.showPopover({ source: trigger });
            } catch {
              menu.showPopover();
            }
          }
        });
      }
    } else {
      // Manual fallback for older browsers without the Popover API.
      const closeManual = ({ restoreFocus = false } = {}) => {
        if (!menu.classList.contains(":popover-open")) return;
        menu.classList.remove(":popover-open");
        setExpanded(false);
        if (restoreFocus) {
          trigger.focus({ focusVisible: false });
        }
      };

      trigger.addEventListener("click", (event) => {
        event.stopPropagation();
        const opening = !menu.classList.contains(":popover-open");
        if (opening) {
          menu.classList.add(":popover-open");
          setExpanded(true);
          positionMenuFallback(trigger, menu);
        } else {
          closeManual();
        }
      });

      document.addEventListener("click", (event) => {
        if (!root.contains(event.target)) {
          closeManual();
        }
      });

      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && menu.classList.contains(":popover-open")) {
          closeManual({ restoreFocus: true });
        }
      });
    }

    if (!hasAnchorPos) {
      window.addEventListener(
        "resize",
        () => {
          const isOpen = hasNativePopover
            ? menu.matches(":popover-open")
            : menu.classList.contains(":popover-open");
          if (isOpen) positionMenuFallback(trigger, menu);
        },
        { passive: true }
      );
    }

    menu.querySelectorAll("a[data-lang]").forEach((link) => {
      link.addEventListener("click", () => {
        try {
          localStorage.setItem(STORAGE_KEY, link.dataset.lang);
        } catch {
          // Ignore storage restrictions (e.g., strict file:// or privacy mode).
        }
      });
    });
  }

  function initAll() {
    document.querySelectorAll(".lang-switcher").forEach(initSwitcher);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
