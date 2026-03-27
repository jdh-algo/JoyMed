"""
Markdown-JSON Converter and Field Name Matcher for Medical Lab Test Tables

This module provides robust conversion between markdown tables and JSON format,
specifically designed for medical/lab test data with flexible field name matching.
"""

import re
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class MatchingResult:
    """Result of field name matching."""

    matched_pairs: List[Tuple[str, str]]  # (source_field, target_field)
    unmatched_source: List[str]
    unmatched_target: List[str]
    confidence: float  # 0.0 to 1.0


def markdown_to_json(markdown_text: str) -> List[Dict[str, str]]:
    """
    Convert markdown table to JSON format.

    Args:
        markdown_text: Markdown table string

    Returns:
        List of dictionaries, each representing a row

    Raises:
        ValueError: If table format is severely malformed
    """
    if not markdown_text or not isinstance(markdown_text, str):
        return []

    # Extract markdown table from code blocks if present
    markdown_patterns = [
        r"```markdown\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]

    for pattern in markdown_patterns:
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        for match in matches:
            if "|" in match:
                return _parse_markdown_table(match)

    # If no code blocks, try parsing the entire text
    if "|" in markdown_text:
        return _parse_markdown_table(markdown_text)

    return []


def _is_separator_line(line: str) -> bool:
    """Check if a line is a markdown table separator line."""
    # Remove pipes and spaces
    cleaned = line.replace("|", "").strip()

    # If empty after removing pipes, it's likely a separator
    if not cleaned:
        return True

    # Check if the line consists mostly of dashes, colons, and spaces
    # These are the typical characters in separator lines
    separator_chars = set(["-", ":", " ", "\t"])
    line_chars = set(cleaned)

    # If all characters (except pipes) are separator characters, it's a separator line
    if line_chars.issubset(separator_chars):
        return True

    # Parse the line into parts separated by pipes
    parts = [p.strip() for p in line.split("|")]
    parts = [p for p in parts if p]  # Remove empty parts

    # Check if all parts are separator-like
    is_separator = True
    for part in parts:
        part_stripped = part.strip()

        # Skip empty parts
        if not part_stripped:
            continue

        # Remove colons (for :---: style separators)
        part_no_colons = part_stripped.replace(":", "").strip()

        # Check if the part consists only of dashes (any number of them)
        # This handles any length: -, --, ---, ----, etc.
        if part_no_colons and not all(c == "-" for c in part_no_colons):
            # This part contains non-dash characters, so it's not a separator
            is_separator = False
            break

    # Additional pattern check: if line contains at least 2 consecutive dashes
    # and all non-whitespace, non-pipe characters are dashes or colons
    if is_separator or re.search(r"-{2,}", cleaned):
        # Double-check by looking at the overall pattern
        # Remove pipes, spaces, tabs, and colons
        check_str = re.sub(r"[|\s:]+", "", line)
        # If what remains is only dashes (or empty), it's a separator
        if not check_str or all(c == "-" for c in check_str):
            return True

    return is_separator


def _parse_markdown_table(markdown_text: str) -> List[Dict[str, str]]:
    """Parse markdown table into JSON format with better handling of escaped content."""
    # Clean up escaped characters first
    markdown_text = markdown_text.replace("\\n", "\n")
    markdown_text = markdown_text.replace("\\t", "\t")
    markdown_text = markdown_text.replace("\\r", "")

    lines = markdown_text.strip().split("\n")
    result = []
    headers = None
    max_columns = 0
    found_separator = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line contains pipes
        if "|" not in line:
            continue

        # Parse table row
        parts = [part.strip() for part in line.split("|")]

        # Remove empty parts at beginning and end
        while parts and not parts[0]:
            parts.pop(0)
        while parts and not parts[-1]:
            parts.pop(-1)

        # Skip if no meaningful content
        if not parts:
            continue

        if headers is None:
            # This is the header row
            headers = parts
            max_columns = len(headers)
            # Clean up headers - remove any special characters that might have been escaped
            headers = [h.strip() for h in headers]
        elif not found_separator and _is_separator_line(line):
            # This is the separator line - skip it but mark that we found it
            found_separator = True
            continue
        else:
            # This is a data row
            # But double-check it's not a separator line (in case of malformed tables)
            if _is_separator_line(line):
                continue

            # Additional check: if all parts consist only of dashes (any number), skip
            all_dashes = True
            for part in parts:
                part_clean = part.strip()
                if part_clean and not all(c == "-" for c in part_clean):
                    all_dashes = False
                    break

            if all_dashes and any(part.strip() for part in parts):
                # If all non-empty parts are just dashes, skip this row
                continue

            if len(parts) != max_columns:
                # Pad with empty strings or truncate
                if len(parts) < max_columns:
                    parts.extend([""] * (max_columns - len(parts)))
                else:
                    parts = parts[:max_columns]

            # Create row dictionary
            row = {}
            for i, header in enumerate(headers):
                if i < len(parts):
                    # Clean up the value - remove brackets and extra spaces
                    value = parts[i].strip()
                    # Skip if value is just dashes (any number of them)
                    if value and all(c == "-" for c in value):
                        continue
                    # Handle values that might have brackets like [HBsAg]
                    # But keep the full value
                    row[header] = value
                else:
                    row[header] = ""

            # Only add row if it has at least one non-empty value that's not just dashes
            has_valid_content = any(v and not all(c == "-" for c in v) for v in row.values())
            if has_valid_content:
                result.append(row)

    return result


def json_to_markdown(json_data: List[Dict[str, str]]) -> str:
    """
    Convert JSON data back to markdown table format.

    Args:
        json_data: List of dictionaries representing table rows

    Returns:
        Markdown table string
    """
    if not json_data:
        return ""

    # Get all unique field names (union of all fields)
    all_fields = set()
    for row in json_data:
        all_fields.update(row.keys())

    # Preserve original field order from the first row
    field_names = []
    if json_data:
        # Use the first row's field order as the template
        field_names = list(json_data[0].keys())
        # Add any missing fields from other rows
        for row in json_data[1:]:
            for field in row.keys():
                if field not in field_names:
                    field_names.append(field)

    if not field_names:
        return ""

    # Build markdown table
    lines = []

    # Header row
    header_line = "| " + " | ".join(field_names) + " |"
    lines.append(header_line)

    # Separator row
    separator_line = "| " + " | ".join([":--:" for _ in field_names]) + " |"
    lines.append(separator_line)

    # Data rows
    for row in json_data:
        row_values = [row.get(field, "") for field in field_names]
        data_line = "| " + " | ".join(row_values) + " |"
        lines.append(data_line)

    return "\n".join(lines)


def match_field_names(
    source_fields: List[str],
    target_fields: List[str],
    keyword_groups: List[List[str]],
    llm_model=None,
    use_llm_fallback: bool = True,
) -> MatchingResult:
    """
    Match field names between source and target using keyword groups and optional LLM.

    Args:
        source_fields: List of source field names
        target_fields: List of target field names
        keyword_groups: List of keyword groups for matching
        llm_model: Optional LLM model for fallback matching
        use_llm_fallback: Whether to use LLM when rule-based matching is incomplete

    Returns:
        MatchingResult with matched pairs and unmatched fields
    """
    matched_pairs = []
    unmatched_source = source_fields.copy()
    unmatched_target = target_fields.copy()

    # Step 1: Rule-based matching using keyword groups
    for group_idx, group in enumerate(keyword_groups):
        # Find all source and target fields that match this group
        matching_source = []
        matching_target = []

        for field in unmatched_source:
            field_lower = field.lower()
            if any(keyword.lower() in field_lower for keyword in group):
                matching_source.append(field)

        for field in unmatched_target:
            field_lower = field.lower()
            if any(keyword.lower() in field_lower for keyword in group):
                matching_target.append(field)

        # Create matches (1-to-1 mapping, prefer exact matches)
        used_source = set()
        used_target = set()

        # First, try exact matches
        for s_field in matching_source:
            for t_field in matching_target:
                if s_field.lower() == t_field.lower():
                    if s_field not in used_source and t_field not in used_target:
                        matched_pairs.append((s_field, t_field))
                        used_source.add(s_field)
                        used_target.add(t_field)

        # Then, try partial matches
        for s_field in matching_source:
            if s_field in used_source:
                continue
            for t_field in matching_target:
                if t_field in used_target:
                    continue
                # Check if they share any keyword
                s_lower = s_field.lower()
                t_lower = t_field.lower()
                if any(keyword.lower() in s_lower and keyword.lower() in t_lower for keyword in group):
                    matched_pairs.append((s_field, t_field))
                    used_source.add(s_field)
                    used_target.add(t_field)
                    break
                # Also check if one field contains the other
                elif s_lower in t_lower or t_lower in s_lower:
                    matched_pairs.append((s_field, t_field))
                    used_source.add(s_field)
                    used_target.add(t_field)
                    break
                # If they're both in the same group, they should match
                elif s_field in matching_source and t_field in matching_target:
                    matched_pairs.append((s_field, t_field))
                    used_source.add(s_field)
                    used_target.add(t_field)
                    break

        # Update unmatched lists
        unmatched_source = [f for f in unmatched_source if f not in used_source]
        unmatched_target = [f for f in unmatched_target if f not in used_target]

    # Step 2: LLM fallback if requested and available
    if use_llm_fallback and llm_model and (unmatched_source or unmatched_target):
        llm_matches = _llm_field_matching(unmatched_source, unmatched_target, llm_model)
        matched_pairs.extend(llm_matches)

        # Update unmatched lists
        for s_field, t_field in llm_matches:
            if s_field in unmatched_source:
                unmatched_source.remove(s_field)
            if t_field in unmatched_target:
                unmatched_target.remove(t_field)

    # Calculate confidence (percentage of fields matched)
    total_fields = len(source_fields) + len(target_fields)
    matched_fields = len(matched_pairs) * 2
    confidence = matched_fields / total_fields if total_fields > 0 else 0.0

    return MatchingResult(
        matched_pairs=matched_pairs,
        unmatched_source=unmatched_source,
        unmatched_target=unmatched_target,
        confidence=confidence,
    )


def _llm_field_matching(source_fields: List[str], target_fields: List[str], llm_model) -> List[Tuple[str, str]]:
    """
    Use LLM to match remaining field names.

    Args:
        source_fields: Unmatched source fields
        target_fields: Unmatched target fields
        llm_model: LLM model for matching

    Returns:
        List of matched field pairs
    """
    if not source_fields or not target_fields:
        return []

    prompt = f"""请帮我匹配以下两组字段名称，这些字段来自医学检验报告表格。

源字段: {', '.join(source_fields)}
目标字段: {', '.join(target_fields)}

请分析这些字段的含义，找出语义上匹配的字段对。考虑以下因素：
1. 字段的医学含义
2. 字段的缩写和全称
3. 字段的常见表达方式
4. 字段的功能和用途

请以JSON格式返回匹配结果，格式如下：
{{
    "matches": [
        {{"source": "源字段名", "target": "目标字段名", "confidence": "high/medium/low"}},
        ...
    ],
    "unmatched_source": ["未匹配的源字段"],
    "unmatched_target": ["未匹配的目标字段"]
}}

只返回JSON，不要其他解释。"""

    try:
        msgs = [{"type": "text", "value": prompt}]
        response = llm_model.generate(msgs)

        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            matches = []
            for match in result.get("matches", []):
                source = match.get("source")
                target = match.get("target")
                if source in source_fields and target in target_fields:
                    matches.append((source, target))
            return matches
    except Exception as e:
        print(f"LLM matching failed: {e}")

    return []


def test_conversion_roundtrip():
    """Test markdown ↔ JSON conversion roundtrip."""
    test_cases = [
        # Standard 4-column format with separator line containing ----
        """| 检验项目 | 结果 | 单位 | 参考范围 |
| ---- | ---- | ---- | ---- |
| 甲型流感病毒抗原检测 | 阴性（-） | 无 | 阴性 |
| 乙型流感病毒抗原检测 | 阴性（-） | 无 | 阴性 |""",
        # With varying dash counts
        """| 检验项目 | 结果 | 单位 | 参考范围 |
| - | -- | --- | -------- |
| 血糖 | 5.2 | mmol/L | 3.9-6.1 |""",
        # Another problematic case
        """| 检验项目 | 结果 | 单位 | 参考范围 |
| ---- | ---- | ---- | ---- |
| 人绒毛膜促性腺激素 | <0.50 | mIU/ml | 0 - 10 |
| 孕酮 | 1.24 | ng/Ml | 无（原表未给出） |""",
        # Mixed separator styles
        """| 项目 | 数值 | 参考值 |
| :-----: | :--: | :--------: |
| 白细胞 | 6.5 | 3.5-9.5 |
| 红细胞 | 4.8 | 4.3-5.8 |""",
        # Extreme case with many dashes
        """| Test | Value | Unit | Reference |
| ------------------------ | ----- | ------------ | --- |
| Glucose | 5.2 | mmol/L | 3.9-6.1 |
| Cholesterol | 4.5 | mmol/L | <5.2 |""",
        # Standard format with :--:
        """| 检验项目 | 结果 | 单位 | 参考范围 |
| :--: | :--: | :--: | :--: |
| 血糖 | 5.2 | mmol/L | 3.9-6.1 |
| 胆固醇 | 4.5 | mmol/L | <5.2 |""",
        # 6-column format
        """| 序号 | 检验项目 | 结果 | 单位 | 参考范围 | 状态 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 1 | 血糖 | 5.2 | mmol/L | 3.9-6.1 | 正常 |
| 2 | 胆固醇 | 4.5 | mmol/L | <5.2 | 正常 |""",
    ]

    print("=== Testing Markdown ↔ JSON Conversion ===")

    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print("Original Markdown:")
        print(test_case)

        # Convert to JSON
        json_data = markdown_to_json(test_case)
        print(f"\nJSON Data ({len(json_data)} rows):")
        print(json.dumps(json_data, ensure_ascii=False, indent=2))

        # Check for the bug - make sure no entries are just dashes
        has_separator_bug = False
        for row in json_data:
            for k, v in row.items():
                if v and isinstance(v, str) and all(c == "-" for c in v):
                    has_separator_bug = True
                    print(f"⚠️ Found dash-only value: '{v}' in field '{k}'")

        if has_separator_bug:
            print("⚠️ WARNING: Found dash-only values in parsed data - separator line was incorrectly parsed as data!")

        # Convert back to markdown
        markdown_back = json_to_markdown(json_data)
        print(f"\nConverted Back to Markdown:")
        print(markdown_back)

        # Verify roundtrip
        json_back = markdown_to_json(markdown_back)
        is_same = json_data == json_back
        print(f"\nRoundtrip Test: {'✅ PASS' if is_same else '❌ FAIL'}")
        print(f"Bug Check: {'✅ No separator bug' if not has_separator_bug else '❌ Separator bug detected'}")
        print("=" * 80)


def test_field_matching():
    """Test field name matching functionality."""
    print("\n=== Testing Field Name Matching ===")

    # Test data
    source_fields = ["检验项目", "结果", "单位", "参考范围", "备注"]
    target_fields = ["entryname", "result", "unit", "reference", "notes"]

    keyword_groups = [
        ["检验项目", "检查项目", "项目", "项目名称", "entryname", "test_name"],
        ["结果", "检测结果", "result", "value", "test_result"],
        ["单位", "unit", "measurement_unit"],
        ["参考范围", "参考值", "reference", "normal_range", "reference_range"],
        ["备注", "notes", "comment", "remark"],
    ]

    # Test without LLM
    result = match_field_names(source_fields, target_fields, keyword_groups, use_llm_fallback=False)

    print(f"Source fields: {source_fields}")
    print(f"Target fields: {target_fields}")
    print(f"Keyword groups: {keyword_groups}")
    print(f"\nMatching result:")
    print(f"Matched pairs: {result.matched_pairs}")
    print(f"Unmatched source: {result.unmatched_source}")
    print(f"Unmatched target: {result.unmatched_target}")
    print(f"Confidence: {result.confidence:.2f}")

    print(f"\nTest: {'✅ PASS' if result.confidence == 1.0 else '❌ FAIL'}")


if __name__ == "__main__":
    test_conversion_roundtrip()
    test_field_matching()
